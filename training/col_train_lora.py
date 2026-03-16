import argparse
import math
import os
import resource
from contextlib import nullcontext
from functools import partial
from typing import Optional, Union

from datasets import concatenate_datasets, load_from_disk

import torch
import torch.nn as nn

from peft import LoraConfig, TaskType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from col_data_utils import prepare_dataloader
from colToolkit import Toolkit, Trainer, WandbConfig
from utils.train_utils import (
    ensure_hybrid_parallel_compatibility,
    format_numel_str,
    get_model_numel,
    get_parallel_ranks,
    patch_qwen2_rotary_embedding_forward,
)


def tokenize_batch_for_finetune(
    batch, tokenizer=None,
    max_length: int = 2048, ring_attn: bool = False
):
    data = tokenizer(
        [sample["input"] for sample in batch],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    if ring_attn:
        data["attention_mask"] = torch.ones_like(data["attention_mask"])

    data["labels"] = data["input_ids"].clone()
    data = {k: v.cuda() for k, v in data.items()}
    return data


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


class LoRATrainer(Trainer):
    def save(self, epoch: int, step: int, batch_size: int):
        super().save(epoch, step, batch_size)
        save_dir = os.path.join(self.save_dir, f"epoch{epoch}-step{step}", "adapter")
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self.booster, "save_lora_as_pretrained"):
            self.booster.save_lora_as_pretrained(self.model, save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="pretrained checkpoint path")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["zero2", "zero2_cpu", "hybrid_parallel"],
        default="hybrid_parallel",
        help="Choose which plugin to use",
    )
    parser.add_argument("-d", "--dataset", type=str, default="yizhongw/self_instruct", help="Data set path")
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Dataset split to use. If empty and a DatasetDict is loaded, all splits are concatenated.",
    )
    parser.add_argument("--task_name", type=str, default="super_natural_instructions", help="task to run")
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Local batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-w", "--weigth_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("-x", "--mixed_precision", default="bf16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument(
        "-i", "--save_interval", type=int, default=None, help="Save interval in steps; default saves at each epoch end"
    )
    parser.add_argument("-o", "--save_dir", type=str, default="checkpoint_lora", help="Checkpoint directory")
    parser.add_argument("-f", "--load", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("-a", "--flash_attention", action="store_true", help="Use Flash Attention")
    parser.add_argument("--ppsize", default=2, type=int)
    parser.add_argument("--tpsize", default=4, type=int)
    parser.add_argument("--spsize", type=int, default=1, help="Sequence parallel size")
    parser.add_argument("--microbatch_size", default=2, type=int)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("-seed", "--shuffle_seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--use_wandb", action="store_true", help="Log training metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity")
    parser.add_argument("--wandb_group", type=str, default=None, help="Weights & Biases group name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument(
        "--sp_mode",
        default="all_to_all",
        choices=["all_to_all", "ring_attn", "ring", "split_gather"],
        help="Sequence parallelism mode",
    )
    parser.add_argument("--lora_rank", "--lora", dest="lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--lora_bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="",
        help="Comma separated module suffixes. Empty uses a default projection-module list.",
    )
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

    tp_rank, dp_rank, is_last_stage, use_pipeline = get_parallel_ranks(booster)
    if use_pipeline:
        print_flag = is_last_stage and tp_rank == 0 and dp_rank == 0
    else:
        print_flag = coordinator.is_master()
    should_log_wandb = print_flag

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(args.dataset)
    if hasattr(dataset, "keys"):
        split_names = list(dataset.keys())
        if args.split:
            if args.split not in dataset:
                raise ValueError(
                    f"Requested split '{args.split}' was not found in dataset {args.dataset}. "
                    f"Available splits: {split_names}"
                )
            coordinator.print_on_master(
                f"Warning: dataset at {args.dataset} is a DatasetDict. Using split '{args.split}' only."
            )
            train_ds = dataset[args.split]
        else:
            coordinator.print_on_master(
                f"Warning: dataset at {args.dataset} is a DatasetDict and no split was provided. "
                f"Concatenating all splits: {split_names}."
            )
            train_ds = concatenate_datasets([dataset[split_name] for split_name in split_names])
    else:
        coordinator.print_on_master(f"Warning: dataset at {args.dataset} is a single Dataset. Using it directly.")
        train_ds = dataset

    train_ds = train_ds.shuffle(seed=args.shuffle_seed)
    dataloader = prepare_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pptp_size=args.pptp_size,
        collate_fn=partial(
            tokenize_batch_for_finetune,
            tokenizer=tokenizer,
            max_length=args.max_length,
            ring_attn=("ring" in args.sp_mode),
        ),
    )

    torch_dtype = get_torch_dtype(args.mixed_precision)
    config = AutoConfig.from_pretrained(args.model_path)
    init_ctx = LazyInitContext(default_device=get_current_device()) if args.plugin == "hybrid_parallel" else nullcontext()
    target_modules: Union[str, list[str]]

    with init_ctx:
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="flash_attention_2" if args.flash_attention else "eager",
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
        model = booster.enable_lora(model, lora_config=lora_config)
        if args.plugin == "hybrid_parallel":
            model = ensure_hybrid_parallel_compatibility(model)

    model.train()
    if args.grad_checkpoint:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()

    if len(dataloader) == 0:
        raise ValueError("The dataloader is empty. Check dataset size, batch_size, and drop_last settings.")

    coordinator.print_on_master(f"LoRA target modules: {target_modules}")
    coordinator.print_on_master(f"Model params: {format_numel_str(get_model_numel(model))}")
    coordinator.print_on_master(f"Train params: {format_numel_str(get_model_numel(model, True))}")

    optimizer = HybridAdam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weigth_decay,
    )
    total_step = args.num_epochs * len(dataloader) // args.grad_accum
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=total_step, warmup_steps=math.ceil(total_step * 0.03), eta_min=0.1 * args.lr
    )

    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )
    torch.set_default_dtype(torch.float)

    booster.load_model(model, args.model_path)

    coordinator.print_on_master(f"Boosted train params: {format_numel_str(get_model_numel(model, True))}")
    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
    )

    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    model_name = os.path.basename(os.path.normpath(args.model_path))

    wandb_config = WandbConfig(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        run_name=args.wandb_run_name if args.wandb_run_name else f"{model_name}-lora",
        config={
            "dataset": dataset_name,
            "batch_size": args.batch_size,
            "micro_batch_size": args.microbatch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": target_modules if isinstance(target_modules, str) else ",".join(target_modules),
        },
    )

    trainer = LoRATrainer(
        booster=booster,
        coordinator=coordinator,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        vocab_size=tokenizer.vocab_size,
        save_dir=args.save_dir,
        print_flag=print_flag,
        should_log_wandb=should_log_wandb,
        wandb_config=wandb_config,
    )
    if args.load is not None:
        coordinator.print_on_master("Loading checkpoint")
        start_epoch, start_step, _ = trainer.load(args.load)
        coordinator.print_on_master(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")

    trainer.training_loop(
        toolkit=Toolkit,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        print_flag=print_flag,
        grad_accum=args.grad_accum,
        use_pipeline=use_pipeline,
    )


if __name__ == "__main__":
    main()
