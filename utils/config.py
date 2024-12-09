from typing import Self
from pydantic import BaseModel, ConfigDict
import yaml

def load_yaml_file(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.Loader)

# Base Class for configs based on dataclasses
# allows for loading multiple configs from multiple yaml files and merging them
class BaseConfig(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def load_from_files(cls, files: list[str], default: dict = {}) -> "Self":
        configs = [
            load_yaml_file(file) for file in files
        ]
        # now merge them to a single config, starting with lowest priority
        merged_config = default
        for config in configs:
            if config is not None:  
                merged_config.update(config)
        return cls(**merged_config)
