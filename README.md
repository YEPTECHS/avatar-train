# Avatar-Train

# Data Preparation

## AWS S3 Dataset Path

**Bucket:** digital-human-llm/

**Object Path:** dev/train-data/aligned-frames/HDTF-####/######.png

**IMAGES HTTP URL:** <https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/dev/train-data/aligned-frames/HDTF-####/######.png>

**AUDIO HTTP URL:** <https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/dev/train-data/audio>[/HDTF-####/######.npy](https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/dev/train-data/aligned-frames/HDTF-####/######.npy)

**Metadata:** <https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/dev/train-data/metadata.npz>"

**Metadata Format:**

    {"folder_data": [
        {'folder_name': 'HDTF_0001', 'image_count': 751, 'image_names': ['000000', '000001', '000002',...]},
        {'folder_name': 'HDTF_0002', 'image_count': 751, 'image_names': ['000000', '000001', '000002',...]},
    ]}

## Train Configuration

configuration files in `configuration/` directory

## Run

1.  Make sure you have installed `uv`

2.  Run `uv run python3 train.py` in the root directory of the project

