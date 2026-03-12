import argparse
import math
import os
import resource
from functools import partial
from typing import Optional

from datasets import load_from_disk

import torch
import torch.nn as nn

from transformers import AutoTokenizer
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam

from col_data_utils import prepare_dataloader
from col_lora import convert_to_lora_module
from colToolkit import Toolkit, Trainer, WandbConfig


def get_model_numel(model: nn.Module, filter_: bool = False) -> int:
    if filter_:
        return sum(p.numel() for p in filter(lambda x: x.requires_grad, model.parameters()))
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def tokenize_batch_for_finetune(
    batch, tokenizer: Optional[LlamaTokenizer] = None,
    max_length: int = 2048, ring_attn: bool = False
):
    data = tokenizer(
        [sample['input'] for sample in batch],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    if ring_attn:
        data["attention_mask"] = torch.ones_like(data["attention_mask"])

    data["labels"] = data["input_ids"].clone()
    # if ring_attn:
    #     data["labels"] = torch.where(
    #         data["attention_mask"] > prompt["attention_mask"],
    #         data["input_ids"], -100)
    #     data["attention_mask"] = torch.ones_like(data["attention_mask"])
    # else:
    #     data["labels"] = data["input_ids"].clone()
    data = {k: v.cuda() for k, v in data.items()}
    return data

def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="pretrained checkpoint path, used with mode==finetune")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["zero2", "zero2_cpu", "hybrid_parallel"],
        default="hybrid_parallel",
        help="Choose which plugin to use",
    )
    parser.add_argument("-d", "--dataset", type=str, default="yizhongw/self_instruct", help="Data set path")
    parser.add_argument("--task_name", type=str, default="super_natural_instructions", help="task to run")
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Local batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-w", "--weigth_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("-x", "--mixed_precision", default="bf16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("-i", "--save_interval", type=int, default=None, help="Save interval in steps; default saves at each epoch end")
    parser.add_argument("-o", "--save_dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("-f", "--load", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("-a", "--flash_attention", action="store_true", help="Use Flash Attention")
    parser.add_argument("--ppsize", default=2, type=int)
    parser.add_argument("--tpsize", default=4, type=int)
    parser.add_argument("--spsize", type=int, default=1, help="Sequence parallel size")
    parser.add_argument("--microbatch_size", default=2, type=int)
    parser.add_argument("--lora", default=0, type=int)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("-seed", "--shuffle_seed", type=int, default=37, help="Shuffle seed")
    parser.add_argument("--use_wandb", action="store_true", help="Log training metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument(
        "--sp_mode",
        default="all_to_all",
        choices=["all_to_all", "ring_attn", "ring", "split_gather"],
        help="Sequence parallelism mode",
    )
    parser.add_argument("--gemma", action="store_true", help="Use gemma model architecture")
    args = parser.parse_args()

    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0.")

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Booster
    # ==============================
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
        # modify the param accordingly, default configuration is for llama2-7b
        # args.ppsize, args.tpsize = 2, 4
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
            enable_metadata_cache=False
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_stage)

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config_class = LlamaConfig if not args.gemma else Gemma2Config
    model_class = LlamaForCausalLM if not args.gemma else Gemma2ForCausalLM

    config = config_class.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.mixed_precision == 'bf16' else
                torch.float16 if args.mixed_precision == 'fp16' else None,
        attn_implementation="flash_attention_2" if args.flash_attention else "eager"
    )

    model = model_class(config)
    if args.lora != 0:
        model = convert_to_lora_module(model, args.lora)

    # ==============================
    # Initialize Tokenizer, Dataset and Dataloader
    # ==============================
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    # follows fast chat: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L257
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(args.dataset)
    if hasattr(dataset, "keys"):
        if "train" not in dataset:
            raise ValueError(
                f"Loaded dataset dict from {args.dataset}, but no 'train' split was found."
            )
        coordinator.print_on_master(
            f"Warning: dataset at {args.dataset} is a dataset dict, using dataset['train'] only."
        )
        train_ds = dataset["train"]
    else:
        train_ds = dataset
    train_ds = train_ds.shuffle(seed=args.shuffle_seed)
    dataloader = prepare_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pptp_size=args.pptp_size,
        collate_fn=partial(
            tokenize_batch_for_finetune, tokenizer=tokenizer,
            max_length=args.max_length, ring_attn=('ring' in args.sp_mode)
        ),
    )

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    if len(dataloader) == 0:
        raise ValueError(
            "The dataloader is empty. Check dataset size, batch_size, and drop_last settings."
        )

    model_numel = get_model_numel(model, True)
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")

    optimizer = HybridAdam(filter(lambda x: x.requires_grad, model.parameters()), 
                           lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weigth_decay)
    total_step = args.num_epochs * len(dataloader) // args.grad_accum
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=total_step, warmup_steps=math.ceil(total_step * 0.03), eta_min=0.1 * args.lr
    )
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )
    
    coordinator.print_on_master(f"Train params: {format_numel_str(get_model_numel(model, True))}")
    
    torch.set_default_dtype(torch.float)

    booster.load_model(model, args.model_path)

    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )

    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    model_name = os.path.basename(os.path.normpath(args.model_path))

    wandb_config = WandbConfig(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name if args.wandb_run_name else model_name,
        config={
            "dataset": dataset_name,
            "batch_size": args.batch_size,
            "micro_batch_size": args.microbatch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.lr,
        },
    )

    trainer = Trainer(
        booster=booster,
        coordinator=coordinator,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        vocab_size=tokenizer.vocab_size,
        save_dir=args.save_dir,
        print_flag=print_flag,
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
