import yaml

from typing import Literal
from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""
    learning_rate: float = Field(default=5e-5, description="Initial learning rate")
    scale_lr: bool = Field(
        default=False,
        description="Scale learning rate by GPUs, gradient accumulation steps, and batch size",
    )
    lr_scheduler: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = Field("cosine")
    lr_warmup_steps: int = Field(
        default=0, description="Number of warmup steps in lr scheduler"
    )
    adam_beta1: float = Field(0.9)
    adam_beta2: float = Field(0.999)
    adam_weight_decay: float = Field(1e-2)
    adam_epsilon: float = Field(1e-8)
    max_grad_norm: float = Field(1.0)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OptimizerConfig":
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)