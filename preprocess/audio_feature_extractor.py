from pathlib import Path

import numpy as np
from tqdm import tqdm

from .submodules.whisper.audio2feature import Audio2Feature


class AudioFeatureExtractor:
    def __init__(self, data, output_path):
        '''
        Args:
            data: List of audio paths
            output_path: Path to the output directory
        '''
        self.data = data
        self.output_path: Path = Path(output_path)
        self.model = Audio2Feature()

    def run(self):
        for audio_path in tqdm(self.data):
            self._extract(audio_path)

    def _extract(self, audio_path):
        whisper_feature = self.model.audio2feat(str(audio_path))
        for __ in range(0, len(whisper_feature) - 1,
                        2):  # -1 to avoid index error if the list has an odd number of elements
            # Combine two consecutive chunks
            # pair_of_chunks = np.array([whisper_feature[__], whisper_feature[__+1]])
            concatenated_chunks = np.concatenate([whisper_feature[__], whisper_feature[__ + 1]], axis=0)
            file_name = image_name = '%06d' % ((__ // 2)) + ".npy"
            save_path =  self.output_path / 'audio' / audio_path.stem / file_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the pair to a .npy file
            np.save(str(save_path), concatenated_chunks)


