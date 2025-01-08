import yaml

from typing import Literal
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Train configuration"""

    gradient_accumulation_steps: int = Field(
        default=512, description="Gradient accumulation steps"
    )
    mixed_precision: Literal["fp16", "bf16"] = Field(
        default="fp16", description="Mixed precision training, fp16 or bf16"
    )
    wandb_enabled: bool = Field(default=False, description="Use wandb for logging")
    gradient_checkpointing: bool = Field(
        default=False, description="Gradient checkpointing for large models"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    dataset_split_ratio: float = Field(
        default=0.8, description="Ratio of train/val split for dataset"
    )
    max_train_steps: int = Field(
        default=100000, description="Maximum number of training steps"
    )
    validation_steps: int = Field(
        default=100, description="Number of validation steps per epoch"
    )
    checkpoints_total_limit: int = Field(
        default=100, description="Total number of checkpoints to keep"
    )
    train_batch_size: int = Field(default=8, description="Training batch size")
    audio_length_left: int = Field(default=2, description="Left context length")
    audio_length_right: int = Field(default=2, description="Right context length")
    finetune: bool = Field(default=False, description="Finetune the model")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainConfig":
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
