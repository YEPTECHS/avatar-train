from typing import Dict
from pydantic import BaseModel

from .data import DataConfig
from .model import ModelConfig
from .optimizer import OptimizerConfig
from .train import TrainConfig


class Config(BaseModel):
    """Main configuration class"""
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    train: TrainConfig

    @classmethod
    def from_yaml_files(cls, config_paths: Dict[str, str]) -> "Config":
        """
        Load configuration from multiple YAML files
        
        Args:
            config_paths: Configuration file path dictionary, format: {"model": "model.yaml", "data": "data.yaml", ...}
        """
        configs = {
            "model": ModelConfig.from_yaml(config_paths["model"]),
            "data": DataConfig.from_yaml(config_paths["data"]),
            "optimizer": OptimizerConfig.from_yaml(config_paths["optimizer"]),
            "train": TrainConfig.from_yaml(config_paths["train"])
        }
        return cls(**configs)
    

config: Config = Config.from_yaml_files({
    "model": "configuration/global/model.yaml",
    "data": "configuration/global/data.yaml",
    "optimizer": "configuration/global/optimizer.yaml",
    "train": "configuration/global/train.yaml"
})