from typing import Tuple
import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import ProcessGroup
from torch.nn import CrossEntropyLoss
from colossalai.shardformer.shard import ShardConfig
import torch.nn.functional as F


class DistCrossEntropy(Function):
    r"""
    Overwrite the forward and backward function to calculate the cross entropy loss before gather

    Args:
        Function (:class:`torch.autograd.Function`): default
    """

    @staticmethod
    def forward(
        ctx,
        vocab_logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        process_group: ProcessGroup,
        vocab_size: int,
        dtype=torch.float32,
        ref_logps: dict = None,
        micro_batch_rank: int = -1,
    ):
        r"""
        Calculate the cross entropy loss before gather, the origin loss function is as follows:
        loss = -log(exp(x[class])/sum(exp(x[i]))
        and can be rewrite as:
        loss = log(sum(exp(x[i])) - x[class]

        To avoid the `nan` of log(sum(exp(x[i]))), we minus the max of x[i]

        Args:
            vocab_logits (:class:`torch.Tensor`): The logits of the vocabulary, shape is
              [batch_size, seq_len, vocab_size]
            target (:class:`torch.Tensor`): The labels of the vocabulary, shape is
              [batch_size, seq_len]

        Returns:
            :class:`torch.Tensor`: The cross entropy loss
        """
        # print(f"vocab_logits: {vocab_logits.shape}, target: {target.shape}")
        # get the max
        logits_max = torch.max(vocab_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)

        # minus the max to avoid the result of sum of exp is too large and the log is nan
        vocab_logits = vocab_logits - logits_max.unsqueeze(dim=-1)

        # mask the target in the local device
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        if vocab_size == None:
            partition_vocab_size = vocab_logits.size()[-1]
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
        mask = (target < down_threshold) | (target >= up_threshold)
        masked_target = target.clone() - down_threshold
        masked_target[mask] = 0

        # reshape the logits and target
        # reshape the vocab_logits to [bath_size * seq_len, vocab_size]
        # reshape the labels to [bath_size * seq_len]
        self_vocab_size = vocab_logits.size()[-1]
        logits_2d = vocab_logits.view(-1, self_vocab_size)
        masked_target_1d = masked_target.view(-1)

        # extract the x[class] and set the x[other device] to zero
        pred_logits_1d = logits_2d[
            torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device), masked_target_1d
        ]
        pred_logits_1d = pred_logits_1d.clone().contiguous()
        pred_logits = pred_logits_1d.view_as(target)
        pred_logits[mask] = 0.0

        # allreduce the get all x(i,y)
        dist.all_reduce(pred_logits, op=dist.ReduceOp.SUM, group=process_group)
        exp_logits = vocab_logits
        torch.exp(vocab_logits, out=exp_logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1, dtype=torch.float32)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=process_group)

        # calculate the loss
        # loss = log(sum(exp(x[i]))) - x[class]
        loss = torch.where(target == ignore_index, 0.0, torch.log(sum_exp_logits) - pred_logits)
        num_non_zero = torch.sum(loss != 0.0, dim=-1)   # Modify
        ctx.inv_num_non_zero = 1.0 / num_non_zero.sum() # Modify
        loss = torch.sum(loss, dim=-1).div_(num_non_zero)  # Modify

        # calculate the softmax
        exp_logits = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1)).to(dtype)
        exp_logits[target == ignore_index] = 0.0
        ctx.save_for_backward(exp_logits, mask, masked_target_1d)
        ctx.dtype = dtype

        # print(f"loss:: {loss}")

        # return loss.mean()

        num_pr_pair = target.shape[0] >> 1
        all_logps = -loss * num_non_zero
        policy_chosen = all_logps[:num_pr_pair]
        rcl_num_non_zero = num_non_zero[:num_pr_pair]
        policy_rejected = all_logps[num_pr_pair:]
        rrl_num_non_zero = num_non_zero[num_pr_pair:]

        if "ref_chosen_logps" in ref_logps and "ref_rejected_logps" in ref_logps:
            assert (
                0 <= micro_batch_rank < len(ref_logps["ref_chosen_logps"])
            ), f"mircro_batch_rank{micro_batch_rank} dismatch len(ref_logps['ref_chosen_logps']){len(ref_logps['ref_chosen_logps'])}"

            losses, chosen_rewards, rejected_rewardes = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_logps["ref_chosen_logps"][micro_batch_rank],
                ref_logps["ref_rejected_logps"][micro_batch_rank],
            )
            return losses.mean()

        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve the saved tensors
        grad_output = grad_output * ctx.inv_num_non_zero
        exp_logits, mask, masked_target_1d = ctx.saved_tensors

        # use exp logits as the input grad
        grad_logits = exp_logits
        partion_vocab_size = grad_logits.shape[-1]
        grad_logits_2d = grad_logits.view(-1, partion_vocab_size)

        update = 1.0 - mask.view(-1).float().to(ctx.dtype)
        grad_logits_2d[torch.arange(0, grad_logits_2d.shape[0]), masked_target_1d] -= update

        grad_logits.mul_(grad_output.unsqueeze(dim=-1))
        return grad_logits, None, None, None, None, None, None, None


class DistLogprobs(Function):
    r"""
    Overwrite the forward and backward function to calculate the cross entropy loss before gather

    Args:
        Function (:class:`torch.autograd.Function`): default
    """

    @staticmethod
    def forward(
        ctx,
        vocab_logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        process_group: ProcessGroup,
        vocab_size: int,
        dtype=torch.float32,
    ):
        r"""
        Calculate the cross entropy loss before gather, the origin loss function is as follows:
        loss = -log(exp(x[class])/sum(exp(x[i]))
        and can be rewrite as:
        loss = log(sum(exp(x[i])) - x[class]

        To avoid the `nan` of log(sum(exp(x[i]))), we minus the max of x[i]

        Args:
            vocab_logits (:class:`torch.Tensor`): The logits of the vocabulary, shape is
              [batch_size, seq_len, vocab_size]
            target (:class:`torch.Tensor`): The labels of the vocabulary, shape is
              [batch_size, seq_len]

        Returns:
            :class:`torch.Tensor`: -Logprobs, shape=[batch_size x seq_len]
        """
        # print(f"vocab_logits: {vocab_logits.shape}, target: {target.shape}")
        # get the max
        logits_max = torch.max(vocab_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)

        # minus the max to avoid the result of sum of exp is too large and the log is nan
        vocab_logits = vocab_logits - logits_max.unsqueeze(dim=-1)

        # mask the target in the local device
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        if vocab_size == None:
            partition_vocab_size = vocab_logits.size()[-1]
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
        mask = (target < down_threshold) | (target >= up_threshold)
        masked_target = target.clone() - down_threshold
        masked_target[mask] = 0

        # reshape the logits and target
        # reshape the vocab_logits to [bath_size * seq_len, vocab_size]
        # reshape the labels to [bath_size * seq_len]
        self_vocab_size = vocab_logits.size()[-1]
        logits_2d = vocab_logits.view(-1, self_vocab_size)
        masked_target_1d = masked_target.view(-1)

        # extract the x[class] and set the x[other device] to zero
        pred_logits_1d = logits_2d[
            torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device), masked_target_1d
        ]
        pred_logits_1d = pred_logits_1d.clone().contiguous()
        pred_logits = pred_logits_1d.view_as(target)
        pred_logits[mask] = 0.0

        # allreduce the get all x(i,y)
        dist.all_reduce(pred_logits, op=dist.ReduceOp.SUM, group=process_group)
        exp_logits = vocab_logits
        torch.exp(vocab_logits, out=exp_logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1, dtype=torch.float32)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=process_group)

        # calculate the loss
        # loss = log(sum(exp(x[i]))) - x[class]
        loss = torch.where(target == ignore_index, 0.0, torch.log(sum_exp_logits) - pred_logits)
        # loss = torch.sum(loss, dim=-1).div_(num_non_zero)  # Modify

        # calculate the softmax
        exp_logits = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1)).to(dtype)
        exp_logits[target == ignore_index] = 0.0
        ctx.save_for_backward(exp_logits, mask, masked_target_1d)
        ctx.dtype = dtype

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve the saved tensors
        exp_logits, mask, masked_target_1d = ctx.saved_tensors

        # use exp logits as the input grad
        grad_logits = exp_logits
        partion_vocab_size = grad_logits.shape[-1]
        grad_logits_2d = grad_logits.view(-1, partion_vocab_size)

        update = 1.0 - mask.view(-1).float().to(ctx.dtype)
        grad_logits_2d[torch.arange(0, grad_logits_2d.shape[0]), masked_target_1d] -= update

        grad_logits_2d.mul_(grad_output.unsqueeze(dim=-1))
        return grad_logits, None, None, None, None, None


def dpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        loss_type: str = "sigmoid",
        beta: float = 0.1,
        label_smoothing: float = 0
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        loss_type, beta, label_smoothing = (
            loss_type, beta, label_smoothing)
        if loss_type == "sigmoid":
            losses = (
                - F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        elif loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        elif loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * beta)) ** 2
        elif loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        # print(f"policy_chosen_logps = {policy_chosen_logps}, reference_chosen_logps = {reference_chosen_logps}")
        chosen_rewards = (
            beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        # print(f"losses = {losses}, chosen_rewards = {chosen_rewards}, rejected_rewards = {rejected_rewards}")
        return losses, chosen_rewards, rejected_rewards



def default_criterion(outputs, inputs):
    return outputs.loss

    
def prm_criterion(outputs, inputs, beta = 0.9,
        ignore: int = -100,
        label_smoothing: bool = False,
        adversarial_temp: float = -1.0,
        special_token_id = 12902,
    ):
    
    logits = outputs.logits
    input_ids = inputs["input_ids"]
    scores = inputs["scores"]
    weights = inputs["weights"]
    labels = input_ids
    
    # special_token_id = tokenizer.encode('\u043a\u0438', add_special_tokens=False)

    # assert labels_pos[0].shape[0] != (scores != score_pad_token).sum().item(), f"{labels_pos[0].shape[0]} != {(scores != -0.1).sum().item()}"

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    labels_pos = (shift_labels == special_token_id)
    shift_logits = shift_logits[labels_pos]
    shift_labels = shift_labels[labels_pos]

    # Flatten the tokens
    special_labels = shift_labels.view(-1).to(shift_labels.device)
    new_vocab_size = logits.shape[-1]
    special_logits = shift_logits.view(-1, new_vocab_size)
    scores = scores.view(-1)
    weights = weights.view(-1)

    assert special_labels.shape[0] == scores.shape[0], f"Labels' shape {special_labels.shape[0]} != Scores shape {scores.shape[0]}"

    loss = beta_entropy_loss(
        shift_logits=special_logits,
        shift_labels=special_labels, 
        beta=scores,
        ignore=ignore, label_smoothing=label_smoothing, adversarial_temp=adversarial_temp,
    )
    return loss


def beta_entropy_loss(
        logits, labels, beta: torch.tensor,
        ignore: int = -100,
        label_smoothing: bool = False,
        adversarial_temp: float = -1.0,
    ):
    # _lab = labels * (labels != ignore)
    # probs = F.softmax(logits, dim=-1).gather(-1, _lab.unsqueeze(-1)).squeeze(-1)
    # probs[labels == ignore] = beta
    # if not label_smoothing:
    #     probs = torch.where(probs < beta, probs, beta)

    # beta = torch.tensor(beta).to(logits.device)
    # bias = - beta * torch.log(beta) - (1 - beta) * torch.log(1 - beta)
    # loss = - beta * torch.log(probs) - (1 - beta) * torch.log(1 - probs)

    # if adversarial_temp > 0:
    #     adv_weight = F.softmax(adversarial_temp * loss, dim=-1).detach()
    #     return (loss * adv_weight).sum()
    # print(f"\np_max:{probs.max()},\np_min:{probs.min()},\nloss:{loss.shape}, {loss.max(), loss.min(), loss.mean()}\nfilter:{(probs < beta).shape}", flush=True)
    # return loss.mean()
    _lab = labels * (labels != ignore)
    probs = F.softmax(logits, dim=-1).gather(-1, _lab.unsqueeze(-1)).squeeze(-1)
    loss = - beta * torch.log(probs) - (1 - beta) * torch.log(1 - probs)
    return loss.mean()