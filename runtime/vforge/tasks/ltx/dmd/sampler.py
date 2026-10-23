"""Equal-block-count batches across data-parallel ranks without cache sidecars."""

from collections import defaultdict

import torch
from torch.utils.data import Sampler


class BlockCountSampler(Sampler[int]):
    """Every global batch has the same number of causal forward collectives.

    Epoch and cursor belong to the sampler and are checkpointed by torchdata.
    Each DMD phase owns an independent instance through the common stream loader.
    """

    def __init__(self, counts, *, num_replicas, rank, seed=0, shuffle=True, drop_last=True):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid data-parallel sampler topology")
        self.buckets = defaultdict(list)
        for index, count in enumerate(counts):
            if count < 1:
                raise ValueError("rollout block counts must be positive")
            self.buckets[int(count)].append(index)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.cursor = 0
        if len(self) == 0:
            raise ValueError("no complete LTX DMD bucket; use drop_last=false or more samples")

    def _indices(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for count in sorted(self.buckets):
            bucket = self.buckets[count]
            order = (
                torch.randperm(len(bucket), generator=rng).tolist()
                if self.shuffle
                else range(len(bucket))
            )
            indices = [bucket[index] for index in order]
            remainder = len(indices) % self.num_replicas
            if remainder:
                if self.drop_last:
                    indices = indices[:-remainder]
                else:
                    indices += [
                        indices[i % len(indices)] for i in range(self.num_replicas - remainder)
                    ]
            batches.extend(
                indices[i : i + self.num_replicas]
                for i in range(0, len(indices), self.num_replicas)
            )
        order = (
            torch.randperm(len(batches), generator=rng).tolist()
            if self.shuffle
            else range(len(batches))
        )
        return [batches[i][self.rank] for i in order]

    def __iter__(self):
        indices = self._indices()
        while self.cursor < len(indices):
            index = indices[self.cursor]
            self.cursor += 1
            yield index
        self.cursor = 0

    def __len__(self):
        extra = 0 if self.drop_last else self.num_replicas - 1
        return sum((len(bucket) + extra) // self.num_replicas for bucket in self.buckets.values())

    def set_epoch(self, epoch):
        if self.epoch != epoch:
            self.epoch, self.cursor = int(epoch), 0

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.epoch, self.cursor = int(state["epoch"]), int(state["cursor"])
        if not 0 <= self.cursor <= len(self):
            raise ValueError("LTX sampler cursor exceeds this dataset")
