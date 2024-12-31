import os
import cv2
import logging
import tempfile
import requests
import torch

import numpy as np

from typing import Optional, Dict
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def download_and_load_npz(metadata_url: str, save_path="temp_file.npz") -> NDArray:
    try:
        # Download file
        response = requests.get(metadata_url, stream=True)
        response.raise_for_status()  # Check HTTP request success

        # Save to local
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info(f"File downloaded to {save_path}")

        # Load .npz file
        data = np.load(save_path, allow_pickle=True)
        # Optional: Delete temporary file
        os.remove(save_path)

        return data["folder_data"]
    except requests.exceptions.RequestException as e:
        logger.info(f"Error downloading the file: {e}")
        return None
    except Exception as e:
        logger.info(f"Error loading the .npz file: {e}")
        return None


def download_image_from_s3_and_load_to_nparray(url: str) -> NDArray:
    '''Download image from S3 and load it to numpy array.
    
    Args:
        url (str): S3 URL of the image.
    
    Returns:
        NDArray: numpy array of the image, RGB format.
    '''
    try:
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check HTTP request success

        # Save to local
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as file:
            with open(file.name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            img = cv2.imread(file.name)
            if img is None:
                raise Exception(f"Failed to load image from url {url}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading the file: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error loading the image: {e}")
        raise e


def download_audio_from_s3_and_load_to_nparray(url: str) -> Optional[NDArray]:
    try:
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check HTTP request success

        # Save to local
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as file:
            with open(file.name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            data = np.load(file.name)
            return data

    except Exception as e:
        logger.warning(f"Error loading the audio: {e}")
        return None
        

def load_image_from_local_storage_to_nparray(path: str) -> NDArray:
    try:
        img = cv2.imread(path)
        if img is None:
            raise Exception(f"Failed to load image from url {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logger.error(f"Error loading the image from local storage {path}: {e}") 
        raise e
    

def load_audio_from_local_storage_to_nparray(path: str) -> Optional[NDArray]:
    try:  
        data = np.load(path)
        return data

    except Exception as e:
        logger.warning(f"Error loading the audio: {e}")
        return None


def load_dict_from_local_storage_to_nparray(path: str) -> Optional[Dict]:
    try:
        data = np.load(path, allow_pickle=True).item()
        return data
    except Exception as e:
        logger.warning(f"Error loading the features: {e}")
        return None