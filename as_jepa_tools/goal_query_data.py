"""Independent training-goal references, never minibatch negatives.

Each item owns its reference draw. Matching the future offset and image pipeline
keeps source classification from exploiting a different marginal frame sampler.
The source label does not mean a reference goal is physically unreachable.
"""
import hashlib
from torch.utils.data import Dataset


class QueryReferenceWindows(Dataset):
    def __init__(self, dataset, seed, mode='global'):
        if mode not in {'global','episode'}:
            raise ValueError('Reference distribution must be global or episode-conditioned')
        self.dataset, self.reference_seed = dataset, int(seed)
        self.reference_mode = mode

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def reference_index(self, index):
        digest = hashlib.blake2b(f'{self.reference_seed}:{int(index)}:independent-goal'.encode(), digest_size=8).digest()
        draw = int.from_bytes(digest, 'little')
        if self.reference_mode == 'global':
            reference = draw % self.dataset.natural_length
        else:
            import numpy as np
            current = int(index) % self.dataset.natural_length
            slot = int(np.searchsorted(self.dataset.ends,current,side='right'))
            count = int(self.dataset.counts[slot])
            beginning = int(self.dataset.ends[slot])-count
            # Independent uniform window in this episode. Do not exclude the
            # factual window: source collisions are legitimate, not failures.
            reference = beginning + draw % count
        if getattr(self.dataset, 'encode_epoch_index', False):
            reference += (int(index) // self.dataset.natural_length) * self.dataset.natural_length
        return reference

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        reference_index = self.reference_index(index)
        reference = self.dataset[reference_index]
        item.update(reference_pixels=reference['pixels'][3:].clone(),
                    reference_episode=reference['episode'], reference_start=reference['start'],
                    reference_index=reference_index)
        return item
