import numpy as np
from tqdm import tqdm
from pathlib import Path

# Read metadata file
metadata_path = Path('~/scratch4/kuizong/Avatar-Train/preprocess/metadata.npy').expanduser()
metadata = np.load(metadata_path, allow_pickle=True).item()

num = 0 

for k, v in tqdm(metadata.items(), desc="Checking lengths"):
    num += len(v)

print(f"Total number of frames: {num}")