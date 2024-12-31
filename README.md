# Avatar-Train

## Data Preparation

There are two ways to prepare the data:

1. **Download from AWS S3 (Optional)**
   - The dataset is available in AWS S3 bucket: `digital-human-llm/`
   - Image data path: `dev/train-data/aligned-frames/HDTF-####/######.png`
   - Audio data path: `dev/train-data/audio/HDTF-####/######.npy`

2. **Use Local Data**
   - The training system will automatically read data from local storage if available
   - Organize your local data in the same structure as the S3 bucket

### Data Configuration

The data paths and settings can be configured in `configuration/global/data.yaml`:
```yaml
s3_data_base_url: "https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/dev/train-data/"
local_data_base_url: "/path/to/your/local/data"
metadata_url: "https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/dev/train-data/metadata.npz"
output_dir: "output_root/your-experiment-name"
image_folder_name: "aligned-frames"  # folder name for images
audio_folder_name: "audio"          # folder name for audio features
IMAGE_SIZE: 256                     # target image size
```

### Data Structure

```
local_data_directory/
├── aligned-frames/
│   ├── HDTF-0001/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── HDTF-0002/
│       └── ...
└── audio/
    ├── HDTF-0001/
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    └── HDTF-0002/
        └── ...
```

### Metadata Handling

The metadata file (`metadata.npz`) contains information about the dataset structure. You can:

1. **Use Remote Metadata**
   - The system will automatically download metadata from the S3 URL during training
   - URL: `https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/dev/train-data/metadata.npz`

2. **Generate Local Metadata**
   - Use `scripts/make_metadata.py` to create metadata from your local data:
     ```bash
     # Generate from local directory
     python scripts/make_metadata.py --mode scan --input /path/to/aligned-frames --output metadata.npz
     
     # Or convert from existing JSON
     python scripts/make_metadata.py --mode json --input metadata.json --output metadata.npz
     ```

**Metadata Format:**
```json
{
    "folder_data": [
        {
            "folder_name": "HDTF_0001",
            "image_count": 751,
            "image_names": ["000000", "000001", "000002",...]
        },
        {
            "folder_name": "HDTF_0002",
            "image_count": 751,
            "image_names": ["000000", "000001", "000002",...]
        }
    ]
}
```

## Configuration

The project uses YAML configuration files in the `configuration/global/` directory:

### 1. Training Configuration (`train.yaml`)
```yaml
gradient_accumulation_steps: 512    # Number of steps to accumulate gradients
mixed_precision: "fp16"            # Mixed precision training mode
wandb_enabled: true               # Enable Weights & Biases logging
gradient_checkpointing: true      # Enable gradient checkpointing
max_train_steps: 5000            # Maximum training steps
validation_steps: 100            # Steps between validations
checkpoints_total_limit: 1000    # Maximum number of checkpoints to keep
train_batch_size: 12            # Training batch size
audio_length_left: 2            # Audio context window (left)
audio_length_right: 2          # Audio context window (right)
```

### 2. Model Configuration (`model.yaml`)
```yaml
unet_config_file: "configuration/unet/config.json"  # UNet model configuration
vae_pretrained_model_name_or_path: "madebyollin/taesd"  # Pretrained VAE model
resume_from_checkpoint: "path/to/checkpoint"  # Resume training from checkpoint
```

### 3. Optimizer Configuration (`optimizer.yaml`)
```yaml
learning_rate: 5e-5
scale_lr: false
lr_scheduler: "cosine"        # Learning rate scheduler type
lr_warmup_steps: 0           # Number of warmup steps
adam_beta1: 0.9             # Adam optimizer beta1
adam_beta2: 0.999          # Adam optimizer beta2
adam_weight_decay: 1e-2    # Weight decay
adam_epsilon: 1e-8        # Adam epsilon parameter
max_grad_norm: 1.0       # Maximum gradient norm for clipping
```

## Training

1. Environment Setup
   ```bash
   # Install uv package manager
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --all-extras
   ```

2. Start Training
   ```bash
   uv run python3 train.py
   ```

The training system will:
- Automatically load metadata (from S3 or local)
- Read data from local storage if available
- Fall back to downloading from S3 if local data is not found
- Use mixed precision training (fp16) for better performance
- Enable gradient checkpointing to save memory
- Log training progress to Weights & Biases if enabled

## TODO
- [ ] Dynamically local metadata from local data or S3 url
- [ ] Dynamically load data from local or S3
- [x] Smart repeat data for fine-tuning because of the limited dataset size
