"""Independent training-goal references, never minibatch negatives.

Each item owns its reference draw. Matching the future offset and image pipeline
keeps source classification from exploiting a different marginal frame sampler.
The source label does not mean a reference goal is physically unreachable.
"""
import hashlib
from torch.utils.data import Dataset


class QueryReferenceWindows(Dataset):
    def __init__(self, dataset, seed):
        self.dataset, self.reference_seed = dataset, int(seed)

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def reference_index(self, index):
        digest = hashlib.blake2b(f'{self.reference_seed}:{int(index)}:independent-goal'.encode(), digest_size=8).digest()
        reference = int.from_bytes(digest, 'little') % self.dataset.natural_length
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
