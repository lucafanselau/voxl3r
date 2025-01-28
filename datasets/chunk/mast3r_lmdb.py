# Potential optimization of the mast3r dataset
# - Use LMDB for faster loading

from datasets import scene
from datasets.chunk import image
from torch.utils.data import DataLoader
from utils.basic import get_default_device
from .mast3r import Dataset as Mast3rDataset, Config as Mast3rConfig
import lmdb
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

class Config(Mast3rConfig):
    lmdb_map_size: int = 1099511627776 * 2  # 2TB default map size
    lmdb_batch_size: int = 1000
    access: str = "get"  # 'get' or 'cursor'

class Dataset(Mast3rDataset):
    def __init__(
        self,
        data_config: Config,
        base_dataset: scene.Dataset,
        image_dataset: image.Dataset,
    ):
        super().__init__(data_config, base_dataset, image_dataset)
        self.config: Config
        self.db_path = self.get_lmdb_path()
        self.env = None
        self.keys = []
        self.scene_indices: Dict[str, List[int]] = {}
        self._txn = None
        self.cursor = None
        self.current_idx = 0  # Track cursor position

    def __len__(self):
        return len(self.keys)

    def _init_cursor(self):
        if self.config.access == "cursor" and self.env is not None:
            self._txn = self.env.begin(db=b"chunks")
            self.cursor = self._txn.cursor()
            self.cursor.first()
            self.current_idx = 0

    def get_lmdb_path(self) -> Path:
        data_dir = self.config.mast3r_data_dir or self.config.data_dir
        return Path(data_dir) / self.config.storage_preprocessing / "mast3r_lmdb"

    def check_chunks_exist(self, batch) -> bool:
        """Check if all chunks in batch exist in LMDB"""
        if self.env is None:
            self.load_paths()
        
        with self.env.begin(db=b"metadata") as txn:
            existing_keys = set(pickle.loads(txn.get(b"keys")))
        
        batch_keys = {
            f"{batch['scene_name'][idx]}_{batch['file_name'][idx]}".encode()
            for idx in range(len(batch["scene_name"]))
        }
        return batch_keys.issubset(existing_keys)

    def prepare_data(self):
        if self.config.skip_prepare and not self.config.force_prepare_mast3r:
            self.load_paths()
            self.on_after_prepare()
            self.prepared = True
            return

        # Create LMDB environment
        self.db_path.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(
            str(self.db_path),
            map_size=self.config.lmdb_map_size,
            max_dbs=2,
            writemap=True,
        )

        # Main database for chunks
        chunks_db = env.open_db(b"chunks")
        # Metadata database
        meta_db = env.open_db(b"metadata")

        with env.begin(write=True, db=meta_db) as txn:
            if not self.config.force_prepare_mast3r and txn.get(b"prepared"):
                return

        # Process chunks and write to LMDB
        with env.begin(write=True, db=chunks_db) as txn:
            txn.drop(db=chunks_db)

        
        # NOTE: we should probably not import from experiments here
        from experiments.mast3r_baseline.module import (
            Mast3rBaselineConfig,
            Mast3rBaselineLightningModule,
        )

        model = Mast3rBaselineLightningModule(Mast3rBaselineConfig())
        model.eval()
        model.model.eval()
        model.model.to(get_default_device())

        batch_size = self.data_config.batch_size_mast3r

        dataloader = DataLoader(
            self.image_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
        )

        key_index = 0
        scene_keys = {}
        with env.begin(write=True, db=chunks_db) as txn, \
             env.begin(write=True, db=meta_db) as meta_txn:
            
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                for chunk_dict, _, _ in super().process_chunk(batch, batch_idx, model):
                    # Serialize data
                    key = f"{chunk_dict['scene_name']}_{chunk_dict['file_name']}".encode()
                    value = pickle.dumps(chunk_dict)
                    
                    # Store in LMDB
                    txn.put(key, value)
                    
                    # Track keys and scene indices
                    self.keys.append(key)
                    scene_name = chunk_dict["scene_name"]
                    if scene_name not in scene_keys:
                        scene_keys[scene_name] = []
                    scene_keys[scene_name].append(key_index)
                    key_index += 1

                # Commit periodically
                if batch_idx % self.config.lmdb_batch_size == 0:
                    txn.commit()
                    meta_txn.put(b"keys", pickle.dumps(self.keys))
                    meta_txn.put(b"scene_indices", pickle.dumps(scene_keys))
                    txn = env.begin(write=True, db=chunks_db)
                    meta_txn = env.begin(write=True, db=meta_db)

            # Final commit
            meta_txn.put(b"keys", pickle.dumps(self.keys))
            meta_txn.put(b"scene_indices", pickle.dumps(scene_keys))
            meta_txn.put(b"prepared", b"true")

        env.close()
        self.load_paths()
        self.on_after_prepare()
        self.prepared = True

    def load_paths(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.db_path),
                max_readers=self.config.lmdb_batch_size,
                readonly=True,
                lock=False,
                readahead=True,
            )
            
        with self.env.begin(db=b"metadata") as txn:
            self.keys = pickle.loads(txn.get(b"keys"))
            self.scene_indices = pickle.loads(txn.get(b"scene_indices"))

        self._init_cursor()

    def get_identifiers(self, scenes: List[str] = None) -> List[Tuple[str, str]]:
        identifiers = []
        for scene_name, indices in self.scene_indices.items():
            if scenes and scene_name not in scenes:
                continue
            for idx in indices:
                key = self.keys[idx]
                file_name = key.decode().split("_", 1)[1]
                identifiers.append((scene_name, file_name))
        return identifiers

    def get_at_idx(self, idx: int):
        if self.env is None:
            self.load_paths()

        if self.config.access == "cursor":
            if not self.cursor:
                self._init_cursor()
            if idx != self.current_idx:
                self.cursor.set_key(self.keys[idx])
            self.current_idx = idx + 1
            data = self.cursor.value()
        else:
            with self.env.begin(db=b"chunks") as txn:
                data = txn.get(self.keys[idx])

        return pickle.loads(data) if data else None

    def __del__(self):
        if self.env:
            self.env.close()


    