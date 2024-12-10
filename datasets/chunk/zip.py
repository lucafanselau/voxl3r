from loguru import logger
from typing import Callable, List, Optional
from torch.utils.data import Dataset
from datasets.chunk.base import ChunkBaseDataset

def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__

class ZipChunkDataset(Dataset):
    def __init__(self, datasets: List[ChunkBaseDataset], transform: Optional[Callable] = None):
        self.datasets = datasets
        self.prepared = False
        self.transform = transform

    def prepare_data(self):
        for dataset in self.datasets:
            logger.info(f"Preparing dataset {fullname(dataset)}")
            dataset.prepare_data()

        # Now let's check if all datasets carry the same identifiers
        idents = [d.get_identifiers() for d in self.datasets]
        logger.info(f"Finished preparing datasets, found: {[(fullname(dataset), len(idents)) for dataset, idents in zip(self.datasets, idents)]} identifiers")

        # find all identifiers that are in every dataset, and for each dataset, the ones which are not in the common set
        common_idents = set.intersection(*[set(ids) for ids in idents])
        for dataset, ids in zip(self.datasets, idents):
            not_common_idents = set(ids) - common_idents
            if len(not_common_idents) > 0:
                logger.warning(f"Dataset {fullname(dataset)} has the following identifiers which are not in the common set: {not_common_idents}")

        # store common identifiers together with the dataset index
        self.common_idents = list(common_idents)

        # for every common identifier, store the indicies of the respective datasets
        self.lookup = {i: [ids.index(i) for ids in idents] for i in common_idents}

        self.prepared = True

    def __len__(self):
        if not self.prepared:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return len(self.common_idents)

    def __getitem__(self, idx):
        if not self.prepared:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        ident = self.common_idents[idx]
        dataset_idxs = self.lookup[ident]

        # We always expect to get back dictionaries, so let's merge them into a single one
        data = {}
        for dataset, idx in zip(self.datasets, dataset_idxs):
            data.update(dataset[idx])

        if self.transform:
            data = self.transform(data)

        return data
