import yaml

from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path


class ModelConfig(BaseModel):
    """Model architecture and pretrained weights configuration"""
    unet_config_file: Path = Field(
        ..., 
        description="The configuration of unet file"
    )
    vae_pretrained_model_name_or_path: str = Field(
        ...,
        description="Path to pretrained model or model identifier from huggingface.co/models"
    )
    resume_from_checkpoint: Optional[str] = Field(
        None,
        description="Path to checkpoint file"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)