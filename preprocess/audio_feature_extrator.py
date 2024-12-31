from pathlib import Path

import numpy as np
from submodules.whisper.audio2feature import Audio2Feature
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self):
        self.model = Audio2Feature(model_path="/home/ubuntu/scratch4/kuizong/Avatar-Train/pretrained/whisper/tiny.pt")

    def extract_batch(self, audio_paths: list, output_dir: Path):
        """Extract features from multiple audio files and save to output directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for audio_path in audio_paths:
            self._extract(audio_path, output_dir)

    def _extract(self, audio_path: Path, output_dir: Path):
        """Extract features from a single audio file"""
        whisper_feature = self.model.audio2feat(str(audio_path))
        
        for idx in range(0, len(whisper_feature) - 1, 2):
            concatenated_chunks = np.concatenate([whisper_feature[idx], whisper_feature[idx + 1]], axis=0)
            file_name = f'{idx // 2:06d}.npy'
            save_path = output_dir / 'audio' / audio_path.stem / file_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), concatenated_chunks)
