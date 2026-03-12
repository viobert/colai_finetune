import json
import random
from typing import Iterator, Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from colossalai.accelerator import get_accelerator


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def prepare_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    pptp_size: Optional[int] = None,
    **kwargs,
):
    r"""
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.


    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    pptp_size = 1 if pptp_size is None else pptp_size
    sampler = StatefulDistributedSampler(
        dataset,
        num_replicas=process_group.size() // pptp_size,
        rank=process_group.rank() // pptp_size,
        shuffle=shuffle
    )

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        **_kwargs,
    )


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(
            0, vocab_size, (num_samples, max_length), device=get_accelerator().get_current_device()
        )
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def collate_fn(features, pad_token_id: int = 0, max_len: int = 1024):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = min(max(len_ids), max_len - 1)
    # longest = max_len - 1

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for length, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        # ids = feature["input_ids"][: max_len - 1]
        length = min(length, max_len - 1)
        # ids = (
        #     ids
        #     + [tokenizer.eos_token_id]
        #     + [tokenizer.pad_token_id] * (longest - length)
        # )
        input_ids = feature["input_ids"][:length]
        attention_mask = feature["attention_mask"][:length]
        labels = feature["labels"][:length]

        input_ids = input_ids + [pad_token_id] * (longest - length)
        attention_mask = attention_mask + [0] * (longest - length)
        labels = labels + [-100] * (longest - length)

        input_ids_list.append(torch.LongTensor(input_ids))
        attention_mask_list.append(torch.LongTensor(attention_mask))
        labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.stack(labels_list)
    # print(f"len:{longest}, input_ids: {input_ids}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
