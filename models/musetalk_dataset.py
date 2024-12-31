import os
import random
import torch
import logging

import numpy as np

from urllib.parse import urljoin
from torch.utils.data import Dataset

from models.config import config
from helpers import (
    download_and_load_npz,
    choose_random_outside_range,
    download_image_from_s3_and_load_to_nparray,
    download_audio_from_s3_and_load_to_nparray,
    load_audio_from_local_storage_to_nparray,
    load_dict_from_local_storage_to_nparray,
    load_image_from_local_storage_to_nparray
)


logger = logging.getLogger(__name__)


class MusetalkTrainDataset(Dataset):
    def __init__(self, validation_size=0):
        # sliding window length
        self.audio_length_left = config.train.audio_length_left
        self.audio_length_right = config.train.audio_length_right

        # whisper shape
        self.whisper_feature_H: int = 384
        self.whisper_feature_W: int = 5
        self.whisper_feature_concateW = (
            self.whisper_feature_W
            * 2
            * (self.audio_length_left + self.audio_length_right + 1)
        )
        # self.dataset_metadata: np.ndarray = download_and_load_npz(config.data.metadata_url)[validation_size:]

        # Load original data
        original_metadata = download_and_load_npz(config.data.metadata_url)[validation_size:]
        
        # Conditionally repeat data
        if len(original_metadata) <= 20:
            self.dataset_metadata = np.tile(original_metadata, 100)
            logger.info(f"Dataset metadata repeated 100 times, total videos: {len(self.dataset_metadata)}")
        else:
            self.dataset_metadata = original_metadata
            logger.info(f"Dataset metadata used as-is, total videos: {len(self.dataset_metadata)}")
        
        # Create mapping
        self.dataset_mapping = {}
        self.data_samples_mapping = []
        counter = 0

        for idx, item in enumerate(self.dataset_metadata):
            for i in range(int(item['image_count'])):
                self.dataset_mapping[counter] = idx
                self.data_samples_mapping.append(i)
                counter += 1

        self._length = counter
        
    def _load_audio_features(
        self, video_dataset: dict, source_img_base_name: str
    ) -> torch.Tensor:
        audio_features = []
        feature_shape = None
        current_idx = int(source_img_base_name)

        for offset in range(
            -self.audio_length_left,
            self.audio_length_right + 1,
        ):
            new_idx = current_idx + offset
            frame_name = f"{new_idx:06d}"
            
            '''feat_url = urljoin(
                config.data.s3_data_base_url,
                f"audios/{video_dataset['folder_name']}/{frame_name}.npy",
            )'''
            feat_path = os.path.join(
                config.data.local_data_base_url,
                f"{config.data.audio_folder_name}/{video_dataset['folder_name']}/{frame_name}.npy",
            )

            try:
                feat: np.ndarray = load_audio_from_local_storage_to_nparray(path=feat_path)
                if feature_shape is None:
                    feature_shape = feat.shape
                audio_features.append(feat)
            except Exception as e:
                logger.warning(f"Error loading {feat_path}: {e}")
                audio_features.append(None)

        if feature_shape is None:
            logger.error("No valid audio features found")
            raise ValueError("No valid audio features found")

        for i in range(len(audio_features)):
            if audio_features[i] is None:
                audio_features[i] = np.zeros(feature_shape)

        audio_feature = np.concatenate(audio_features, axis=0)
        audio_feature = audio_feature.reshape(-1, self.whisper_feature_H)

        if audio_feature.shape != (
            self.whisper_feature_concateW,
            self.whisper_feature_H,
        ):
            logger.error(f"Invalid audio feature shape: {audio_feature.shape}")
            raise ValueError(f"Invalid audio feature shape: {audio_feature.shape}")

        return torch.FloatTensor(audio_feature)

    def _reshape_image(self, img):
        #  H x W x 3
        x = np.asarray(img) / 255.0
        x = np.transpose(x, (2, 0, 1))  # Convert to C x H x W
        return x

    def __len__(self):
        return len(self.dataset_metadata)

    def __getitem__(self, idx):
        while True:
            try:
                # Video dataset:HDTF-*
                video_dataset = self.dataset_metadata[idx]
                
                # Map idx to image index in the video dataset
                source_img_idx = random.choice(range(len(video_dataset["image_names"])))

                ref_img_idx = choose_random_outside_range(
                    video_dataset["image_names"], source_img_idx, exclusion_range=5
                )

                source_img_base_name = video_dataset["image_names"][source_img_idx]
                ref_img_base_name = video_dataset["image_names"][ref_img_idx]

                '''# Load images from S3 to numpy arrays in RGB format - Too Slow
                source_img: np.ndarray = download_image_from_s3_and_load_to_nparray(
                    url=urljoin(
                        config.data.s3_data_base_url,
                        f"aligned-frames/{video_dataset['folder_name']}/{source_img_base_name}.png",
                    )
                )

                ref_img: np.ndarray = download_image_from_s3_and_load_to_nparray(
                    url=urljoin(
                        config.data.s3_data_base_url,
                        f"aligned-frames/{video_dataset['folder_name']}/{ref_img_base_name}.png",
                    )
                )'''
                
                source_img: np.ndarray = load_image_from_local_storage_to_nparray(
                    path=os.path.join(
                        config.data.local_data_base_url,
                        f"{config.data.image_folder_name}/{video_dataset['folder_name']}/{source_img_base_name}.png",
                    )
                )

                ref_img: np.ndarray = load_image_from_local_storage_to_nparray(
                    path=os.path.join(
                        config.data.local_data_base_url,
                        f"{config.data.image_folder_name}/{video_dataset['folder_name']}/{ref_img_base_name}.png",
                    )
                )

                # Convert images to C x H x W format
                source_img = self._reshape_image(source_img)
                ref_img = self._reshape_image(ref_img)

                # Create mask with shape H x W with 1s in the top half and 0s in the bottom half
                mask = torch.zeros((ref_img.shape[1], ref_img.shape[2]))
                mask[: ref_img.shape[1] // 2, :] = 1

                # Create masked source image
                masked_source_img = torch.FloatTensor(source_img) * mask

                audio_feature = self._load_audio_features(
                    video_dataset=video_dataset, 
                    source_img_base_name=source_img_base_name
                )

                return ref_img, source_img, masked_source_img, mask, audio_feature

            except Exception as e:
                # Retry until a valid sample is found
                logger.error(f"Error loading sample {idx}: {e}", exc_info=True)
                continue


class MusetalkValDataset(Dataset):
    def __init__(self, validation_size=0):
        # sliding window length
        self.audio_length_left = config.train.audio_length_left
        self.audio_length_right = config.train.audio_length_right

        # whisper shape
        self.whisper_feature_H: int = 384
        self.whisper_feature_W: int = 5
        self.whisper_feature_concateW = (
            self.whisper_feature_W
            * 2
            * (self.audio_length_left + self.audio_length_right + 1)
        )
        if validation_size == 0:
            self.dataset_metadata: np.ndarray = download_and_load_npz(config.data.metadata_url)
        else:
            self.dataset_metadata: np.ndarray = download_and_load_npz(config.data.metadata_url)[:validation_size]
        
        logger.warning(f"Using {len(self.dataset_metadata)} validation samples")
        self.dataset_mapping = {}
        self.data_samples_mapping = []
        counter = 0

        for idx, item in enumerate(self.dataset_metadata):
            for i in range(int(item['image_count'])):
                self.dataset_mapping[counter] = idx
                self.data_samples_mapping.append(i)
                counter += 1

        self._length = counter
        
    def _load_audio_features(
        self, video_dataset: dict, source_img_base_name: str
    ) -> torch.Tensor:
        audio_features = []
        feature_shape = None
        current_idx = int(source_img_base_name)

        for offset in range(
            -self.audio_length_left,
            self.audio_length_right + 1,
        ):
            new_idx = current_idx + offset
            frame_name = f"{new_idx:06d}"
            
            '''feat_url = urljoin(
                config.data.s3_data_base_url,
                f"audios/{video_dataset['folder_name']}/{frame_name}.npy",
            )'''
            feat_path = os.path.join(
                config.data.local_data_base_url,
                f"{config.data.audio_folder_name}/{video_dataset['folder_name']}/{frame_name}.npy",
            )

            try:
                feat: np.ndarray = load_audio_from_local_storage_to_nparray(path=feat_path)
                if feature_shape is None:
                    feature_shape = feat.shape
                audio_features.append(feat)
            except Exception as e:
                logger.warning(f"Error loading {feat_path}: {e}")
                audio_features.append(None)

        if feature_shape is None:
            logger.error("No valid audio features found")
            raise ValueError("No valid audio features found")

        for i in range(len(audio_features)):
            if audio_features[i] is None:
                audio_features[i] = np.zeros(feature_shape)

        audio_feature = np.concatenate(audio_features, axis=0)
        audio_feature = audio_feature.reshape(-1, self.whisper_feature_H)

        if audio_feature.shape != (
            self.whisper_feature_concateW,
            self.whisper_feature_H,
        ):
            logger.error(f"Invalid audio feature shape: {audio_feature.shape}")
            raise ValueError(f"Invalid audio feature shape: {audio_feature.shape}")

        return torch.FloatTensor(audio_feature)

    def _reshape_image(self, img):
        #  H x W x 3
        x = np.asarray(img) / 255.0
        x = np.transpose(x, (2, 0, 1))  # Convert to C x H x W
        return x

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        while True:
            try:
                video_dataset = self.dataset_metadata[self.dataset_mapping[idx]]
                source_img_idx = self.data_samples_mapping[idx]
                ref_img_idx = choose_random_outside_range(
                    video_dataset["image_names"], source_img_idx, exclusion_range=5
                )

                source_img_base_name = video_dataset["image_names"][source_img_idx]
                ref_img_base_name = video_dataset["image_names"][ref_img_idx]
                
                # Load source and ref images
                source_img = load_image_from_local_storage_to_nparray(
                    path=os.path.join(
                        config.data.local_data_base_url,
                        f"{config.data.image_folder_name}/{video_dataset['folder_name']}/{source_img_base_name}.png",
                    )
                )

                ref_img = load_image_from_local_storage_to_nparray(
                    path=os.path.join(
                        config.data.local_data_base_url,
                        f"{config.data.image_folder_name}/{video_dataset['folder_name']}/{ref_img_base_name}.png",
                    )
                )

                # Convert images to C x H x W format
                source_img = self._reshape_image(source_img)
                ref_img = self._reshape_image(ref_img)

                # Create mask with shape H x W with 1s in the top half and 0s in the bottom half
                mask = torch.zeros((ref_img.shape[1], ref_img.shape[2]))
                mask[: ref_img.shape[1] // 2, :] = 1

                # Create masked source image
                masked_source_img = torch.FloatTensor(source_img) * mask

                audio_feature = self._load_audio_features(
                    video_dataset=video_dataset, 
                    source_img_base_name=source_img_base_name
                )

                return ref_img, source_img, masked_source_img, mask, audio_feature

            except Exception as e:
                logger.error(f"Error loading sample {idx}: {e}", exc_info=True)
                continue
