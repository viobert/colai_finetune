import torch.nn as nn
import torch.distributed as dist

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin


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
    if numel >= M:
        return f"{numel / M:.2f} M"
    if numel >= K:
        return f"{numel / K:.2f} K"
    return f"{numel}"


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
