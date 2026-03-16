"""
for Debug only

run:
python test/debug_col_train_lora_pre/standalone_debug.py \
    --model_path /mnt/sdb1/mkx/model/Qwen2.5-7B-Instruct \
    --init_mode config \
    --print_modules 20

"""


import argparse
from dataclasses import dataclass
from typing import Optional, Union

try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:
    missing_pkg = exc.name or "unknown"
    raise SystemExit(
        "Missing dependency: "
        f"{missing_pkg}. Install the standalone requirements first: torch, transformers, peft."
    ) from exc


def get_torch_dtype(mixed_precision: str) -> Optional[torch.dtype]:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "fp32":
        return torch.float32
    return None


def format_numel_str(numel: int) -> str:
    if numel >= 1_000_000_000:
        return f"{numel / 1_000_000_000:.2f}B"
    if numel >= 1_000_000:
        return f"{numel / 1_000_000:.2f}M"
    if numel >= 1_000:
        return f"{numel / 1_000:.2f}K"
    return str(numel)


def get_model_numel(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


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


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@dataclass
class DebugArtifacts:
    tokenizer: Optional[AutoTokenizer]
    config: AutoConfig
    base_model: nn.Module
    lora_model: nn.Module
    target_modules: Union[str, list[str]]
    device: torch.device
    torch_dtype: Optional[torch.dtype]


def build_model_and_lora(args: argparse.Namespace) -> DebugArtifacts:
    device = choose_device(args.device)
    torch_dtype = get_torch_dtype(args.mixed_precision)

    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_path)

    if args.init_mode == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch_dtype,
        )

    target_modules = parse_target_modules(model, args.lora_target_modules)
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
    lora_model = get_peft_model(model, lora_config)
    lora_model.to(device)

    return DebugArtifacts(
        tokenizer=tokenizer,
        config=config,
        base_model=model,
        lora_model=lora_model,
        target_modules=target_modules,
        device=device,
        torch_dtype=torch_dtype,
    )


def collect_sample_linear_modules(model: nn.Module, limit: int) -> list[str]:
    samples = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            samples.append(module_name)
        if len(samples) >= limit:
            break
    return samples


def collect_trainable_parameters(model: nn.Module, limit: int) -> list[str]:
    samples = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            samples.append(f"{name} {tuple(param.shape)}")
        if len(samples) >= limit:
            break
    return samples


def print_debug_snapshot(artifacts: DebugArtifacts, args: argparse.Namespace) -> None:
    model = artifacts.lora_model
    tokenizer = artifacts.tokenizer

    print("=== Standalone debug snapshot ===")
    print(f"model_path: {args.model_path}")
    print(f"init_mode: {args.init_mode}")
    print(f"device: {artifacts.device}")
    print(f"mixed_precision: {args.mixed_precision}")
    print(f"torch_dtype: {artifacts.torch_dtype}")
    print(f"attn_implementation: {args.attn_implementation}")
    print(f"config.model_type: {artifacts.config.model_type}")
    print(f"config.hidden_size: {getattr(artifacts.config, 'hidden_size', 'N/A')}")
    print(f"config.num_hidden_layers: {getattr(artifacts.config, 'num_hidden_layers', 'N/A')}")
    print(f"config.vocab_size: {getattr(artifacts.config, 'vocab_size', 'N/A')}")
    print(f"tokenizer_loaded: {tokenizer is not None}")
    if tokenizer is not None:
        print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
        print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")

    print(f"target_modules: {artifacts.target_modules}")
    print(f"total_params: {format_numel_str(get_model_numel(model))}")
    print(f"trainable_params: {format_numel_str(get_model_numel(model, trainable_only=True))}")
    print(f"training_flag_before_train_call: {model.training}")

    print("\n=== Sample linear modules ===")
    for module_name in collect_sample_linear_modules(artifacts.base_model, args.print_modules):
        print(module_name)

    print("\n=== Sample trainable parameters after LoRA ===")
    for param_name in collect_trainable_parameters(model, args.print_modules):
        print(param_name)


def build_dummy_inputs(
    tokenizer: Optional[AutoTokenizer],
    config: AutoConfig,
    device: torch.device,
    dummy_text: str,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    if tokenizer is not None:
        encoded = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=seq_len)
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        return {key: value.to(device) for key, value in encoded.items()}

    pad_token_id = getattr(config, "pad_token_id", 0) or 0
    vocab_size = getattr(config, "vocab_size", 32000)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    input_ids[:, -1] = pad_token_id
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_dummy_forward(artifacts: DebugArtifacts, args: argparse.Namespace) -> None:
    model = artifacts.lora_model
    model.eval()

    inputs = build_dummy_inputs(
        tokenizer=artifacts.tokenizer,
        config=artifacts.config,
        device=artifacts.device,
        dummy_text=args.dummy_text,
        seq_len=args.dummy_seq_len,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    print("\n=== Dummy forward ===")
    print(f"input_ids.shape: {tuple(inputs['input_ids'].shape)}")
    print(f"logits.shape: {tuple(outputs.logits.shape)}")
    print(f"logits.dtype: {outputs.logits.dtype}")
    print(f"first_token_logits[:8]: {outputs.logits[0, 0, :8].detach().float().cpu().tolist()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone debugger for the pre-model.train() section of col_train_lora.py"
    )
    parser.add_argument("--model_path", required=True, type=str, help="Hugging Face model path or local model dir")
    parser.add_argument(
        "--init_mode",
        default="config",
        choices=["config", "pretrained"],
        help="Use config-only init or load pretrained weights",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used to hold the model",
    )
    parser.add_argument(
        "--mixed_precision",
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Torch dtype used when building the model",
    )
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Attention backend passed to Transformers",
    )
    parser.add_argument("--skip_tokenizer", action="store_true", help="Skip tokenizer loading")
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--lora_bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="",
        help="Comma-separated target suffixes, or all-linear",
    )
    parser.add_argument("--print_modules", default=12, type=int, help="How many sample module names to print")
    parser.add_argument("--run_dummy_forward", action="store_true", help="Run one dummy forward pass")
    parser.add_argument("--dummy_text", default="Debug sample input.", type=str)
    parser.add_argument("--dummy_seq_len", default=16, type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.lora_rank <= 0:
        raise ValueError("lora_rank must be greater than 0.")
    if args.print_modules <= 0:
        raise ValueError("print_modules must be greater than 0.")
    if args.dummy_seq_len <= 0:
        raise ValueError("dummy_seq_len must be greater than 0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    artifacts = build_model_and_lora(args)
    print_debug_snapshot(artifacts, args)

    if args.run_dummy_forward:
        run_dummy_forward(artifacts, args)

    print("\nReached the equivalent state right before `model.train()` in the original script.")


if __name__ == "__main__":
    main()
