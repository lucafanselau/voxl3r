import os
from pathlib import Path
from typing import Optional, Union, List, Any, Dict, Tuple
import io
import gc

import torch
import torchvision
from datasets import scene
from loguru import logger
import lmdb
import pickle
from tqdm import tqdm

from datasets.chunk.pair_matching import PairMatching

from .image import Dataset as ImageDataset, Config as ImageDatasetConfig, Output as ImageOutput

class Config(ImageDatasetConfig):
    pair_matching: str = "first_centered"
    lmdb_map_size: int = 500 * 1e9 # 4TB default
    lmdb_batch_size: int = 8
    access: str = "get"  # 'get' or 'cursor' or 'cursor_forced'

    meminit: bool = True
    max_readers: int = 128

    pair_matching: str = "first_centered"

class Output(ImageOutput):
    images_tensor: torch.Tensor
    pairs_image_names: list[tuple[str, str]]


class Dataset(ImageDataset):
    """
    This dataset simply loads images and metadata from the .pt files produced by the chunk.image.Dataset.
    USING LMDB
    It does not generate or prepare chunks, but rather just reads them and converts images to torch tensors.
    """
    def __init__(self, data_config: Config, base_dataset: scene.Dataset):
        super().__init__(data_config, base_dataset)
        self.config: Config = data_config
        self.db_path = self.get_lmdb_path()
        self.env = None
        self.keys = []
        self.scene_indices: Dict[str, List[int]] = {}
        self.pair_matching = PairMatching[data_config.pair_matching]()

    def __getstate__(self):
        """Prepare for pickling - exclude LMDB environment"""
        state = self.__dict__.copy()
        # Remove the unpicklable LMDB environment
        state['env'] = None
        return state

    def __setstate__(self, state):
        """Restore after unpickling - reload LMDB connection"""
        self.__dict__.update(state)
        # Reinitialize LMDB connection
        self.load_paths()

    def get_lmdb_path(self) -> str:
        return (
            Path(self.config.data_dir)
            / self.config.storage_preprocessing
            / "image_loader_lmdb"
        )

    def prepare_data(self):
        if self.config.skip_prepare and not self.config.force_prepare:
            self.load_paths()
            self.prepared = True
            return
        
        self.drop_db()

        for start, end in [(0, 1500), (1500, 3000), (3000, 4500), (4500, 6000), (6000, 7339)]:
            self.prepare_data_range(start, end)
            import gc
            gc.collect()
        

        self.load_paths()
        self.prepared = True


    def drop_db(self):
        if self.env:
            self.env.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
    def prepare_data_range(self, start, end):

        # Create LMDB environment
        # self.db_path.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(
            str(self.db_path),
            map_size=self.config.lmdb_map_size,
            writemap=True,
            subdir=False,
            readonly=False,
            meminit=True,
            map_async=True,
            sync=True
        )

        # Check if already prepared
        # with env.begin(write=False) as txn:
        #     if txn.get(b"__prepared__") and not self.config.force_prepare:
        #         self.prepared = True
        #         return

        # Clear existing data
        #with env.begin(write=True) as txn:
        #    txn.drop(env.open_db())

        image_dataset = ImageDataset(self.config, self.base_dataset)
        image_dataset.prepare_data()

        # try to load the metadata from previous runs
        with env.begin(write=False) as txn:
            # if keys are existing
            if txn.get(b"__keys__"):
                keys = pickle.loads(txn.get(b"__keys__"))
                scene_indices = pickle.loads(txn.get(b"__scene_indices__"))
            else:
                keys = []
                scene_indices = {}
            

       
       
        
        for idx in tqdm(range(start, end)):
            with env.begin(write=True) as txn:
                chunk_dict = image_dataset.get_at_idx(idx)
                scene_name = chunk_dict["scene_name"]
                
                # Load in uint8 and convert to float32 just before storage
                images_tensor = self._load_images_into_tensor(chunk_dict["images"])
                chunk_dict["images_tensor"] = images_tensor

                # Serialize with torch.save to buffer
                buffer = io.BytesIO()
                torch.save(chunk_dict, buffer)
                value = buffer.getvalue()
                
                # Store and immediately clear references
                key = f"{chunk_dict['scene_name']}_{chunk_dict['image_name_chunk']}".encode()
                txn.put(key, value)
                
                keys.append(key)
                # Manual cleanup
                del images_tensor, chunk_dict, value
                if idx % 10 == 0:  # GC more frequently
                    gc.collect()

                # # Commit more often for larger datasets
                # if idx % (self.config.lmdb_batch_size // 4) == 0:
                #     txn.commit()
                #     env.sync()
                #     txn = env.begin(write=True)
                
                # Track scene indices
                if scene_name not in scene_indices:
                    scene_indices[scene_name] = []
                scene_indices[scene_name].append(key)


        with env.begin(write=True) as txn:
            # Store metadata as special keys
            txn.put(b"__keys__", pickle.dumps(keys))
            txn.put(b"__scene_indices__", pickle.dumps(scene_indices))
            txn.put(b"__prepared__", b"true")
            txn.put(b"__len__", pickle.dumps(len(keys)))

        env.close()

        del env
        del txn

    def get_identifiers(self):
        if not self.keys:
            self.load_paths()

        # should return list of tuples of (scene_name, image_name)
        return [(key.decode().split("_")[0], key.decode().split("_")[1]) for key in self.keys]

    def load_paths(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.db_path),
                readonly=True,
                writemap=True,
                lock=False,
                subdir=False,
                readahead=self.config.access != 'get',
                meminit=self.config.meminit,
                max_readers=self.config.max_readers,
            )

            if self.config.access == 'cursor' or self.config.access == 'cursor_forced':
                self._init_cursor()
            
        with self.env.begin() as txn:
            self.keys = pickle.loads(txn.get(b"__keys__"))
            self.scene_indices = pickle.loads(txn.get(b"__scene_indices__"))

    def check_chunks_exist(self, batch) -> bool:
        if self.env is None:
            self.load_paths()
        
        with self.env.begin() as txn:
            existing_keys = set(pickle.loads(txn.get(b"__keys__")))
        
        batch_keys = {
            f"{batch['scene_name'][idx]}_{batch['file_name'][idx]}".encode()
            for idx in range(len(batch["scene_name"]))
        }
        return batch_keys.issubset(existing_keys)

    def get_at_idx(self, idx: int):
        if self.env is None:
            self.load_paths()

        if self.config.access == 'cursor' or self.config.access == 'cursor_forced':
            # init if not already initialized
            if self.cursor is None:
                self._init_cursor()

            requested_key = self.keys[idx]

            if self.config.access != "cursor_forced" and requested_key != self.cursor.key():
                self.cursor.set_key(requested_key)

            key = self.cursor.key()

            data = self.cursor.value()
            self.cursor.next()
        else:
            with self.env.begin() as txn:
                key = self.keys[idx]
                data = txn.get(key)
        
        if not data:
            raise ValueError(f"Invalid index {idx}")
        
        # torch load from bytesio
        try:
            bytes = io.BytesIO(data)
            chunk_dict = torch.load(bytes)
        except Exception as e:
            logger.warning(f"Error loading data for key {key}: {e}")
            return self.get_at_idx(idx + 1, )
        
        # Add pair matching
        seq_len = len(chunk_dict["images"])
        pair_indices = self.pair_matching(seq_len)
        image_names = [str(Path(name).name) for name in chunk_dict["images"]]
        
        chunk_dict["pairs_indices"] = pair_indices
        chunk_dict["pairs_image_names"] = [
            (image_names[pair_idx[0]], image_names[pair_idx[1]]) 
            for pair_idx in pair_indices
        ]
        chunk_dict["pairwise_prediction"] = chunk_dict["images_tensor"][pair_indices]
        
        return chunk_dict
    
    def _init_cursor(self):
        """Initialize cursor position. This is an optional way to index the LMDB."""
        self._txn = self.env.begin(write=False)
        self.cursor = self._txn.cursor()
        self.cursor.first()
        self.internal_index = 0

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        if self.env:
            self.env.close()

    def _load_images_into_tensor(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """Load images as uint8 to save 4x memory"""
        loaded_images = []
        for img_path in image_paths:
            if not os.path.isfile(img_path):
                continue
            # Keep as uint8 until needed
            img = torchvision.io.read_image(str(img_path)).to(torch.uint8)
            loaded_images.append(img)
        return torch.stack(loaded_images) if loaded_images else torch.empty((0, 3, 0, 0))
