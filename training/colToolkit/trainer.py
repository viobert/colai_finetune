import dataclasses
import os
from datetime import datetime
from functools import partial
from typing import Optional

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator

from col_data_utils import load_json, save_json
from .toolkit import DPORefToolkit, Toolkit

try:
    import wandb
except ImportError:
    wandb = None


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


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor.data
    tensor.div_(dist.get_world_size())
    return tensor


def _criterion(outputs, inputs):
    return outputs.loss


@dataclasses.dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    config: Optional[dict] = None


class Trainer:
    def __init__(
            self, booster: Booster, coordinator: DistCoordinator,
            model, dataloader: DataLoader, optimizer: Optimizer,
            lr_scheduler: _LRScheduler, vocab_size: int, save_dir: str = './save_checkpoint',
            **kwargs
        ):
        self.start_epoch, self.start_step, self.sampler_start_idx = 0, 0, 0
        self.model, self.dataloader = model, dataloader
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.booster, self.coordinator = booster, coordinator
        self.num_steps_per_epoch = len(dataloader)
        self.vocab_size = vocab_size
        self.save_dir = save_dir
        self.print_flag = kwargs.get("print_flag", False)
        self.wandb_config = kwargs.get("wandb_config")
        self.should_log_wandb = False

        if self.wandb_config is None:
            self.wandb_config = WandbConfig()
        if self.wandb_config.enabled:
            if wandb is None:
                raise ImportError("wandb is enabled but the wandb package is not installed.")
            if not self.wandb_config.project:
                raise ValueError("wandb is enabled but no project name was provided.")
            self.should_log_wandb = self.print_flag
            if self.should_log_wandb:
                wandb.init(
                    project=self.wandb_config.project,
                    entity=self.wandb_config.entity,
                    name=self.wandb_config.run_name,
                    dir=self.save_dir,
                    config=self.wandb_config.config,
                )

    def load(self, load_dir: str):
        if load_dir is None:
            self.start_epoch, self.start_step, self.sampler_start_idx = 0, 0, 0
            return 0, 0, 0
        self.booster.load_model(self.model, os.path.join(load_dir, "model"))
        self.booster.load_optimizer(self.optimizer, os.path.join(load_dir, "optimizer"))
        self.booster.load_lr_scheduler(self.lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
        running_states = load_json(os.path.join(load_dir, "running_states.json"))
        self.start_epoch, self.start_step, self.sampler_start_idx = (
            running_states["epoch"], running_states["step"], running_states["sample_start_index"])
        return self.start_epoch, self.start_step, self.sampler_start_idx

    def save(self, epoch: int, step: int, batch_size: int,):
        save_dir = os.path.join(self.save_dir, f"epoch{epoch}-step{step}")
        os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

        self.booster.save_model(self.model, os.path.join(save_dir, "model"), shard=True)
        self.booster.save_optimizer(self.optimizer, os.path.join(save_dir, "optimizer"), shard=True)
        self.booster.save_lr_scheduler(self.lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
        running_states = {
            "epoch": epoch, "step": step,
            "sample_start_index": step * batch_size,
        }
        if self.coordinator.is_master():
            save_json(running_states, os.path.join(save_dir, "running_states.json"))

    def criterion(
        self,
        outputs,
        inputs,
        toolkit: Toolkit = None,
        vocab_size: int = -1,
        process_group: ProcessGroup = None,
        status: list = [],
        args: dict = None,
    ):
        status[0] += 1
        
        if args is None:
            args = {}
        return toolkit.compute_loss(
            outputs, inputs,
            micro_batch_rank = status[0],
            vocab_size=vocab_size,
            process_group=process_group,
            **args
        )

    def training_loop(
            self,
            toolkit=Toolkit, toolkit_kwargs: dict = None,
            num_epochs: int = 1, batch_size: int = 1,
            save_interval: Optional[int] = None,
            print_flag: bool = False,
            grad_accum: int = 1, use_pipeline: bool = False,
    ):
        if save_interval is None:
            save_interval = self.num_steps_per_epoch

        self.dataloader.sampler.set_start_index(self.sampler_start_idx)
        for epoch in range(self.start_epoch, num_epochs):
            self.dataloader.sampler.set_epoch(epoch)
            step_range = range(self.start_step, self.num_steps_per_epoch)
            dataloader_iter = iter(self.dataloader)
            status = [-1]
                
            with tqdm(
                step_range,
                desc=f"Epoch {epoch}",
                disable=not print_flag,
                total=self.num_steps_per_epoch,
                initial=self.start_step,
            ) as pbar:
                for step in pbar:
                    status = [-1]
                    batch = next(dataloader_iter)
                    input = toolkit.get_input(batch)
                    loss = None

                    if use_pipeline:
                        criterion = partial(
                            self.criterion,
                            toolkit=toolkit,
                            vocab_size=self.vocab_size,
                            process_group=self.booster.plugin.shard_config.tensor_parallel_process_group,
                            status=status,
                            args=toolkit_kwargs,
                        )

                        result = self.booster.execute_pipeline(
                            iter([input]),
                            self.model,
                            criterion,
                            self.optimizer,
                            return_loss=True,
                            return_outputs=True,
                        )
                        loss = result['loss']
                    else:
                        output = self.model(**input)
                        loss = toolkit.compute_loss(output, batch)
                        self.booster.backward(loss, self.optimizer)

                    if (step + 1) % grad_accum == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                    if not use_pipeline:
                        all_reduce_mean(loss)
                    if print_flag:
                        pbar.set_postfix({"loss": loss.item()})
                        if self.should_log_wandb:
                            wandb.log(
                                {"train/loss": loss.item()},
                                step=(step + 1) + self.num_steps_per_epoch * epoch,
                            )

                    if save_interval > 0 and (step + 1) % save_interval == 0:
                        self.coordinator.print_on_master(f"Saving checkpoint")
                        self.save(epoch, step + 1, batch_size)
                        self.coordinator.print_on_master(
                            f"Saved checkpoint at epoch {epoch} step {step + 1}")

            self.dataloader.sampler.set_start_index(0)
            self.start_step = 0

        if self.should_log_wandb:
            wandb.finish()
        self.coordinator.print_on_master("Saving checkpoint..")
        self.save(epoch, step + 1, batch_size)
        self.coordinator.print_on_master(
            f"Saved checkpoint at epoch {epoch} step {step + 1}")
        self.coordinator.print_on_master(
            f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


class RefTrainer(Trainer):
    def __init__(self, booster, coordinator, model, dataloader, optimizer, lr_scheduler, vocab_size, save_logp_dir):
        super().__init__(booster, coordinator, model, dataloader, optimizer, lr_scheduler, vocab_size)
        self.save_logp_dir = save_logp_dir

    def load(self, load_dir: str):
        if load_dir is None:
            self.start_epoch, self.start_step, self.sampler_start_idx = 0, 0, 0
            return 0, 0, 0
        self.booster.load_model(self.model, os.path.join(load_dir, "model"))
        self.booster.load_optimizer(self.optimizer, os.path.join(load_dir, "optimizer"))
        self.booster.load_lr_scheduler(self.lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
        running_states = load_json(os.path.join(load_dir, "running_states.json"))
        self.start_epoch, self.start_step, self.sampler_start_idx = (
            running_states["epoch"], running_states["step"], running_states["sample_start_index"])
        return self.start_epoch, self.start_step, self.sampler_start_idx

    def training_loop(
        self,
        toolkit=DPORefToolkit,
        num_epochs: int = 1,
        batch_size: int = 1,
        save_interval: int = -1,
        save_dir: str = None,
        print_flag: bool = False,
        grad_accum: int = 1,
        use_pipeline: bool = False,
    ):
        compute_loss_group_rank0_start_time = datetime.now().strftime('%y%m%d%H%M')
        self.dataloader.sampler.set_start_index(self.sampler_start_idx)
        for epoch in range(self.start_epoch, num_epochs):
            self.dataloader.sampler.set_epoch(epoch)
            step_range = range(self.start_step, self.num_steps_per_epoch)
            dataloader_iter = iter(self.dataloader)

            with tqdm(
                step_range,
                desc=f"Epoch {epoch}",
                disable=not print_flag,
                total=self.num_steps_per_epoch,
                initial=self.start_step,
            ) as pbar:
                for step in pbar:
                    batch = next(dataloader_iter)
                    input = toolkit.get_input(batch)

                    loss = None
                    reference_chosen_logp, reference_rejected_logp = [], []
                    with torch.no_grad():
                        if use_pipeline:
                            result = self.booster.execute_pipeline(
                                iter([input]), self.model,
                                lambda output, _: toolkit.compute_loss(batch, output),
                                optimizer=None, return_loss=True, return_outputs=True
                            )
                            # print(f"result:: {result}")

                            if "outputs" in result and result["outputs"] is not None and "logits" in result["outputs"]:
                                process_group=self.booster.plugin.shard_config.tensor_parallel_process_group

                                logits = result["outputs"]["logits"]
                                shift_logits = logits[..., :-1, :].contiguous()
                                shift_labels = batch["labels"][..., 1:].contiguous()

                                # print(f"shift_logits:: {shift_logits.shape}, shift_labels:: {shift_labels.shape}")
                                # print(f"logits:: {logits}")
                                # shift_labels = shift_labels.view(-1)
                                # shift_labels = shift_labels.to(shift_logits.device)
                                # new_vocab_size = logits.shape[-1]
                                # shift_logits = shift_logits.view(-1, new_vocab_size)

                                loss, num_non_zero = toolkit.post_process(logits=shift_logits,
                                                     labels=shift_labels,
                                                     vocab_size=self.vocab_size,
                                                     process_group=process_group)
                                # print(f"loss:: {loss}, num_non_zero:: {num_non_zero}")

                                all_logps = -loss * num_non_zero
                                reference_chosen_logp = all_logps[:batch_size]
                                rcl_num_non_zero = num_non_zero[:batch_size]
                                reference_rejected_logp = all_logps[batch_size:]
                                rrl_num_non_zero = num_non_zero[batch_size:]

                                # print(f"process_group:: {process_group}",
                                #       f"self.coordinator._local_rank:: {self.coordinator._local_rank}",
                                #       f"self.coordinator._rank:: {self.coordinator._rank}",
                                #       f"dist.get_group_rank:: {dist.get_group_rank(process_group, self.coordinator._rank)}",
                                #       f"dist.get_rank:: {dist.get_rank()}",
                                #       f"dist.get_rank(process_group):: {dist.get_rank(process_group)}")

                                # print(f"rank:: {dist.get_rank(process_group)}")
                                if dist.get_rank(process_group) == 0:
                                    # print(f"reference_chosen_logp:: {reference_chosen_logp}, reference_rejected_logp:: {reference_rejected_logp}")

                                    reference_chosen_logp_list = reference_chosen_logp.tolist()
                                    rcl_num_non_zero_list = rcl_num_non_zero.tolist()
                                    reference_rejected_logp_list = reference_rejected_logp.tolist()
                                    rrl_num_non_zero_list = rrl_num_non_zero.tolist()

                                    df = pd.DataFrame({
                                        'reference_chosen_logp': reference_chosen_logp_list,
                                        "reference_chosen_logp_non0len": rcl_num_non_zero_list,
                                        'reference_rejected_logp': reference_rejected_logp_list,
                                        "reference_rejected_logp_non0len": rrl_num_non_zero_list
                                    })
                                    file_path = self.save_logp_dir + "/logp_" + compute_loss_group_rank0_start_time +  ".csv"
                                    # print(f"file_path:: {file_path}")

                                    if os.path.exists(file_path):
                                        df.to_csv(file_path, mode='a', header=False, index=False)
                                    else:
                                        df.to_csv(file_path, mode='w', header=True, index=False)

                        else:
                            output = self.model(**input)
                            if output is not None:
                                reference_chosen_logp, reference_rejected_logp = toolkit.compute_loss(batch, output)

            self.dataloader.sampler.set_start_index(0)
            self.start_step = 0
