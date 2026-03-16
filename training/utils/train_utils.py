import inspect

import torch
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
            # actual_seq_len = seq_len
            position_ids = torch.arange(actual_seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return original_forward(self, x, position_ids=position_ids, *args, **kwargs)

    forward_with_seq_len_compat._colai_ft_seq_len_compat = True
    Qwen2RotaryEmbedding.forward = forward_with_seq_len_compat
