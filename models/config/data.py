import yaml

from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    """Data configuration"""
    s3_data_base_url: str = Field(None, description="S3 data base url")
    local_data_base_url: str = Field(None, description="Local data base url")
    image_folder_name: str = Field("aligned_with_bg", description="Image folder name")
    audio_folder_name: str = Field("audio", description="Audio folder name")
    metadata_url: str
    output_dir: str
    IMAGE_SIZE: int = 256

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataConfig":
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)