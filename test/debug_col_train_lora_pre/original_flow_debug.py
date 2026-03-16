"""
for Debug only

run:
torchrun --nproc_per_node=2 test/debug_col_train_lora_pre/original_flow_debug.py \
    --model_path /mnt/sdb1/mkx/model/Qwen2.5-7B-Instruct \
    --plugin hybrid_parallel \
    --print_limit 20 > test/debug_col_train_lora_pre/debug.log 2>&1
"""

import argparse
import inspect
from contextlib import nullcontext
from typing import Optional, Union

try:
    import torch
    import torch.nn as nn

    import colossalai
    import torch.distributed as dist
    from colossalai.booster import Booster
    from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin
    from colossalai.cluster import DistCoordinator
    from colossalai.lazy import LazyInitContext
    from colossalai.utils import get_current_device
    from peft import LoraConfig, TaskType
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:
    missing_pkg = exc.name or "unknown"
    raise SystemExit(
        "Missing dependency: "
        f"{missing_pkg}. Install standalone requirements first: torch, transformers, peft, colossalai."
    ) from exc


def get_torch_dtype(mixed_precision: str):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def parse_target_modules(model: nn.Module, modules_arg: Optional[str]) -> Union[str, list[str]]:
    if modules_arg:
        modules_arg = modules_arg.strip()
        if modules_arg == "all-linear":
            return modules_arg
        return [module_name.strip() for module_name in modules_arg.split(",") if module_name.strip()]

    candidate_suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    target_modules = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module_name.endswith(candidate_suffixes):
            target_modules.append(module_name.split(".")[-1])

    return sorted(set(target_modules))


def get_parallel_ranks(booster: Booster):
    tp_rank = 0
    dp_rank = 0
    is_last_stage = False
    use_pipeline = False

    if isinstance(booster.plugin, HybridParallelPlugin):
        plugin = booster.plugin
        use_pipeline = plugin.pp_size > 1
        if use_pipeline:
            is_last_stage = plugin.stage_manager.is_last_stage()

        if hasattr(plugin, "tp_group") and plugin.tp_group is not None:
            tp_rank = dist.get_rank(group=plugin.tp_group)

        if hasattr(plugin, "dp_group") and plugin.dp_group is not None:
            dp_rank = dist.get_rank(group=plugin.dp_group)

    return tp_rank, dp_rank, is_last_stage, use_pipeline


def ensure_hybrid_parallel_compatibility(model: nn.Module) -> nn.Module:
    candidate_models = [model]
    for attr_path in ("model", "base_model", "base_model.model", "model.model"):
        current = model
        valid = True
        for attr_name in attr_path.split("."):
            current = getattr(current, attr_name, None)
            if current is None:
                valid = False
                break
        if valid:
            candidate_models.append(current)

    unique_candidates = []
    seen = set()
    for candidate_model in candidate_models:
        if id(candidate_model) in seen:
            continue
        seen.add(id(candidate_model))
        unique_candidates.append(candidate_model)

    for attr_name in ("embed_tokens", "layers", "norm"):
        source_attr = None
        for candidate_model in unique_candidates:
            if hasattr(candidate_model, attr_name):
                source_attr = getattr(candidate_model, attr_name)
                break
        if source_attr is None:
            continue

        for candidate_model in unique_candidates:
            if not hasattr(candidate_model, attr_name):
                setattr(candidate_model, attr_name, source_attr)

    return model


def patch_qwen2_rotary_embedding_forward() -> None:
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    except ImportError:
        return

    if getattr(Qwen2RotaryEmbedding.forward, "_colai_ft_seq_len_compat", False):
        return

    signature = inspect.signature(Qwen2RotaryEmbedding.forward)
    if "seq_len" in signature.parameters:
        return

    original_forward = Qwen2RotaryEmbedding.forward

    def forward_with_seq_len_compat(self, x, position_ids=None, seq_len=None, *args, **kwargs):
        if position_ids is None and seq_len is not None:
            batch_size = x.shape[0]
            actual_seq_len = x.shape[-2]
            position_ids = torch.arange(actual_seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return original_forward(self, x, position_ids=position_ids, *args, **kwargs)

    forward_with_seq_len_compat._colai_ft_seq_len_compat = True
    Qwen2RotaryEmbedding.forward = forward_with_seq_len_compat


def print_named_modules(model: nn.Module, limit: int) -> None:
    print("\n=== named_modules sample ===")
    count = 0
    for module_name, module in model.named_modules():
        print(f"{module_name}: {type(module).__name__}")
        count += 1
        if count >= limit:
            break


def print_trainable_params(model: nn.Module, limit: int) -> None:
    print("\n=== trainable params sample ===")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={tuple(param.shape)} dtype={param.dtype}")
            count += 1
            if count >= limit:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="pretrained checkpoint path")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["zero2", "zero2_cpu", "hybrid_parallel"],
        default="hybrid_parallel",
        help="Choose which plugin to use",
    )
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Local batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-w", "--weigth_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("-x", "--mixed_precision", default="bf16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("-a", "--flash_attention", action="store_true", help="Use Flash Attention")
    parser.add_argument("--ppsize", default=2, type=int)
    parser.add_argument("--tpsize", default=1, type=int)
    parser.add_argument("--spsize", type=int, default=1, help="Sequence parallel size")
    parser.add_argument("--microbatch_size", default=2, type=int)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("-seed", "--shuffle_seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--sp_mode", default="all_to_all", choices=["all_to_all", "ring_attn", "ring", "split_gather"])
    parser.add_argument("--lora_rank", "--lora", dest="lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--lora_bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--print_limit", type=int, default=20)
    args = parser.parse_args()

    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0.")
    if args.lora_rank <= 0:
        raise ValueError("lora_rank must be greater than 0.")

    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    args.pptp_size = None
    if args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, cpu_offload=True, max_norm=args.grad_clip
        )
    elif args.plugin == "hybrid_parallel":
        patch_qwen2_rotary_embedding_forward()
        args.pptp_size = args.ppsize * args.tpsize
        if args.sp_mode == "ring_attn":
            args.pptp_size *= args.spsize
        plugin = HybridParallelPlugin(
            tp_size=args.tpsize,
            pp_size=args.ppsize,
            sp_size=args.spsize,
            sequence_parallelism_mode=args.sp_mode,
            enable_sequence_parallelism=args.spsize > 1,
            enable_flash_attention=args.flash_attention,
            num_microbatches=None,
            microbatch_size=args.microbatch_size,
            enable_jit_fused=False,
            zero_stage=0,
            precision=args.mixed_precision,
            initial_scale=1,
            enable_metadata_cache=False,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)
    print(f"booster::{booster}")

    tp_rank, dp_rank, is_last_stage, use_pipeline = get_parallel_ranks(booster)
    if use_pipeline:
        print_flag = is_last_stage and tp_rank == 0 and dp_rank == 0
    else:
        print_flag = coordinator.is_master()
    should_log_wandb = print_flag

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # Deliberately skipped from the original script:
    # dataset = load_from_disk(args.dataset)
    # train_ds = ...
    # dataloader = prepare_dataloader(...)
    dataset = None
    train_ds = None
    dataloader = None

    torch_dtype = get_torch_dtype(args.mixed_precision)
    config = AutoConfig.from_pretrained(args.model_path)
    print(f"config:: {config}")
    init_ctx = LazyInitContext(default_device=get_current_device()) if args.plugin == "hybrid_parallel" else nullcontext()
    target_modules: Union[str, list[str]]

    with init_ctx:
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="flash_attention_2" if args.flash_attention else "eager",
            torch_dtype=torch_dtype,
        )
        print(f"model_before lora booster:: {model}")
        target_modules = parse_target_modules(model, args.lora_target_modules)
        print(f"target_modules before:: {args.lora_target_modules}")
        print(f"target_modules after:: {target_modules}")
        if not target_modules:
            raise ValueError("No LoRA target modules were found. Set --lora_target_modules explicitly.")

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        print(f"lora_config: {lora_config}")
        model = booster.enable_lora(model, lora_config=lora_config)
        print(f"model_after lora booster:: {model}")
        if args.plugin == "hybrid_parallel":
            model = ensure_hybrid_parallel_compatibility(model)
        print(f"model_after ensure_hybrid_parallel_compatibility booster:: {model}")

    print("=== debug values before model.train() ===")
    print(f"print_flag={print_flag}")
    print(f"should_log_wandb={should_log_wandb}")
    print(f"tokenizer.pad_token={tokenizer.pad_token}")
    print(f"tokenizer.eos_token={tokenizer.eos_token}")
    print(f"dataset={dataset}")
    print(f"train_ds={train_ds}")
    print(f"dataloader={dataloader}")
    print(f"torch_dtype={torch_dtype}")
    print(f"type(config)={type(config).__name__}")
    print(f"type(model)={type(model).__name__}")
    print(f"target_modules={target_modules}")
    print(f"model.training={model.training}")

    print_named_modules(model, args.print_limit)
    print_trainable_params(model, args.print_limit)

    print("\nReached the exact point immediately before `model.train()`.")


if __name__ == "__main__":
    main()
