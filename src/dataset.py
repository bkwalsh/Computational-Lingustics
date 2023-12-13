import collections
import json
from typing import Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


SPLITS = ['train', 'dev', 'test']


class HateSpeechDataset(Dataset):
    """ Javanese dataset.
    """
    def __init__(self: Dataset, path: str):
        """ Reads the data from the path.

        Parameters
        ----------
        path: Path to the original data
        """
        with open(path) as fin:
            self._data = [json.loads(l) for l in fin]
        self._n_classes = len(set([datum['label'] for datum in self._data]))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    @property
    def n_classes(self):
        return self._n_classes

    @staticmethod
    def collate_fn(tokenizer: AutoTokenizer, device: torch.device,
                   batch: list[dict[str, Any]]) -> Tuple[torch.LongTensor, dict[str, torch.Tensor]]:
        """ The collate function that compresses a training batch.

        Parameters
        ----------
        tokenizer: Model tokenizer that converts sentences to integer tensors
        device: Device (CPU/GPU) that the tensor should be on
        batch: Data in the batch

        Returns
        -------
        labels: Labels in the batch
        sentences: Sentences converted by tokenizers
        """
        labels = torch.tensor([datum['label'] for datum in batch]).long().to(device)
        sentences = tokenizer(
            [datum['sentence'] for datum in batch],
            return_tensors='pt',  # pt = pytorch style tensor
            padding=True)
        for key in sentences:
            sentences[key] = sentences[key].to(device)
        return labels, sentences


def construct_datasets(prefix: str, batch_size: int, tokenizer: AutoTokenizer,
                       device: torch.device) -> Tuple[dict[str, Dataset], dict[str, DataLoader]]:
    """ Constructs datasets and data loaders.

    Parameters
    ----------
    prefix: Prefix of the dataset (e.g., data/fold1/)
    batch_size: Maximum number of examples in a batch
    tokenizer: Model tokenizer that converts sentences to integer tensors
    device: Device (CPU/GPU) that the tensor should be on

    Returns
    -------
    datasets: Dict of constructed datasets
    dataloaders: Dict of constructed data loaders
    """
    datasets = collections.defaultdict()
    dataloaders = collections.defaultdict()
    for split in SPLITS:
        datasets[split] = HateSpeechDataset(f'{prefix}{split}.json')
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            collate_fn=lambda x: HateSpeechDataset.collate_fn(tokenizer, device, x))
    return datasets, dataloaders