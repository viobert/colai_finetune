import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .criterion import DistCrossEntropy, DistLogprobs
from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

class Toolkit:
    @staticmethod
    def get_input(batch):
        return batch
    
    @staticmethod
    def compute_loss(
        outputs,
        *args, **kwargs
    ):
        if isinstance(outputs, dict):
            return outputs['loss']
        return outputs[0]
    
    @staticmethod
    def post_process(batch, logits):
        return logits


class DPORefToolkit(Toolkit):
    loss_type: str = "sigmoid"
    beta: float = 0.1
    label_smoothing: float = 0
    
    @staticmethod
    def get_input(batch):
        return dict(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch.get("labels", None),
                    )
    
    @staticmethod
    def post_process(logits, labels, vocab_size, process_group, ignore_index = -100):
        # print(f"logits1:: {logits.shape}, logits1:: {logits}")
        # if logits.shape[:2] != labels.shape[:2]:
        #     # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
        #     seq_len = labels.shape[1]
        #     logits = logits[:, -seq_len:]
        # print(f"logits2:: {logits.shape}, logits2:: {logits}")

        logits_max = torch.max(logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)

        # minus the max to avoid the result of sum of exp is too large and the log is nan
        logits = logits - logits_max.unsqueeze(dim=-1)

        # mask the target in the local device
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        if vocab_size == None:
            partition_vocab_size = logits.size()[-1]
            global_vocab_size = partition_vocab_size * world_size
        else:
            global_vocab_size = vocab_size
            partition_vocab_size = global_vocab_size // world_size

        # [down, up) => false, other device and -100 => true
        delta = (global_vocab_size + world_size - 1) // world_size
        down_threshold = rank * delta
        up_threshold = down_threshold + delta
        if up_threshold > global_vocab_size:
            up_threshold = global_vocab_size
        mask = (labels < down_threshold) | (labels >= up_threshold)
        masked_target = labels.clone() - down_threshold
        masked_target[mask] = 0

        # reshape the logits and labels
        # reshape the logits to [bath_size * seq_len, vocab_size]
        # reshape the labels to [bath_size * seq_len]
        self_vocab_size = logits.size()[-1]
        logits_2d = logits.view(-1, self_vocab_size)
        masked_target_1d = masked_target.view(-1)

        # extract the x[class] and set the x[other device] to zero
        pred_logits_1d = logits_2d[
            torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device), masked_target_1d
        ]
        pred_logits_1d = pred_logits_1d.clone().contiguous()
        pred_logits = pred_logits_1d.view_as(labels)
        pred_logits[mask] = 0.0

        # allreduce the get all x(i,y)
        dist.all_reduce(pred_logits, op=dist.ReduceOp.SUM, group=process_group)
        exp_logits = logits
        torch.exp(logits, out=exp_logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1, dtype=torch.float32)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=process_group)

        # calculate the loss
        # loss = log(sum(exp(x[i]))) - x[class]
        loss = torch.where(labels == ignore_index, 0.0, torch.log(sum_exp_logits) - pred_logits)
        num_non_zero = torch.sum(loss != 0.0, dim=-1)
        # ctx.inv_num_non_zero = 1.0 / num_non_zero
        loss = torch.sum(loss, dim=-1).div_(num_non_zero)

        # calculate the softmax
        # exp_logits = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1)).to(dtype)
        # exp_logits[labels == ignore_index] = 0.0
        # ctx.save_for_backward(exp_logits, mask, masked_target_1d)
        # ctx.dtype = dtype

        return loss, num_non_zero
    
    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape} and labels {labels.shape} must have the same shape."
            )
        
        len_chosen = labels.shape[0] >> 1

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        all_logps, size_completion = (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        return chosen_logps, rejected_logps


class DPOToolkit(Toolkit):
    loss_type: str = "sigmoid"
    beta: float = 0.1
    label_smoothing: float = 0

    @staticmethod
    def compute_loss(
        outputs,
        inputs,
        micro_batch_rank: int = -1,
        vocab_size: int = 0,
        process_group: ProcessGroup = None,
    ):
        labels = inputs.get("labels", inputs["input_ids"])
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        ref_logps = {
            "ref_chosen_logps": inputs.get("ref_chosen_logps", None),
            "ref_rejected_logps": inputs.get("ref_rejected_logps", None),
        }
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = DistCrossEntropy.apply(
                shift_logits,
                shift_labels,
                -100,
                process_group,
                vocab_size,
                torch.float32,
                ref_logps,
                micro_batch_rank,
        )
        return loss


class PRMToolkit(Toolkit):
    @staticmethod
    def compute_loss(
        outputs,
        inputs,
        micro_batch_rank: int = -1,
        vocab_size: int = 0,
        process_group: ProcessGroup = None,
        ignore: int = -100,
        special_token_id: Optional[int] = 1107,
        # special_token_id = 12902,
    ):
        logits = outputs.logits
        micro_batch_size = logits.shape[0]
        input_ids = inputs["input_ids"]
        scores = inputs["scores"]
        weights = inputs["weights"]
        special_token_pos = inputs.get("special_token_pos", None)
        if special_token_pos is not None:
            pos_mask = torch.ones(weights.shape[0], 1, dtype=torch.long, device=weights.device) * torch.arange(
                weights.shape[1], dtype=torch.long, device=weights.device).unsqueeze(0)
            pos_mask = pos_mask >= special_token_pos.int().sum(dim=1).unsqueeze(1)
            scores[pos_mask] = -1
            weights[pos_mask] = -1

        # ---------- Truncate Process ----------
        # res = []
        # for i, example in enumerate(scores):
        #     truncate_count = (input_ids[i] == special_token_id).sum()
        #     valid_scores = scores[i][:truncate_count]
        #     res.append(valid_scores)
        # scores = torch.cat(res)
        # ---------- Truncate Process ----------
        # assert special_token_pos.int().sum() > 0, "special_token_pos must be greater than 0"

        scores = scores[scores >= 0]
        weights = weights[weights >= 0]
        labels = input_ids
        # import pdb; pdb.set_trace()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if isinstance(special_token_pos, torch.Tensor):
            labels_pos = special_token_pos[..., 1:].contiguous()
        else:
            labels_pos = (shift_labels == special_token_id)
        if scores.shape[0] != labels_pos.sum():
            print('\n-------- FAIL --------------\n')

        shift_logits = shift_logits[labels_pos]
        shift_labels = shift_labels[labels_pos]

        # Flatten the tokens
        special_labels = shift_labels.view(-1).to(shift_labels.device)
        new_vocab_size = logits.shape[-1]
        special_logits = shift_logits.view(-1, new_vocab_size)
        scores = scores.view(-1).to(shift_labels.device)
        weights = weights.view(-1).to(shift_labels.device)

        assert special_labels.shape[0] == scores.shape[0], f"Labels' shape {special_labels.shape[0]} != Scores' shape {scores.shape[0]}"

        logprobs = -DistLogprobs.apply(
            special_logits,
            special_labels,
            ignore,
            process_group,
            vocab_size,
            torch.float32
        )  # DistLogprobs returns MINUS logprobs!
        # logprobs = shift_logits.squeeze(-1)
        if scores.min() > 0.99:
            loss = -logprobs
        else:
            loss = - scores * logprobs - (1 - scores) * torch.log(1 - torch.exp(logprobs))
            base = - scores * torch.log(scores + 1e-8) - (1 - scores) * torch.log(1 - scores + 1e-8)
            loss = loss - base
        # print(f"loss:: {loss}")
        loss = (loss * weights).sum() / weights.sum()
        return loss
