import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process
from typing import Dict, List

def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

class FeatureExtractor:
    def __init__(self, vae, unet, device):
        self.vae = vae
        self.unet = unet
        self.device = device
        self.features = {}
        self.setup_hooks()
    
    def setup_hooks(self):
        """Set hooks for downblock, midblock, and upblock of UNet"""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output[0].detach().cpu().numpy()
            return hook

        # Add hooks for downblock, midblock, and upblock of UNet
        for name, layer in self.unet.named_modules():
            if 'down_blocks' in name.lower() or 'mid_block' in name.lower() or 'up_blocks' in name.lower():
                layer.register_forward_hook(get_activation(f'unet_{name}'))

    def extract_features(self, image_path: str) -> Dict[str, np.ndarray]:
        """Extract features from a single image"""
        self.features.clear()

        # Image preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  # Add resize to ensure correct image size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # VAE encoding
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # UNet inference
            batch_size = latents.shape[0]
            empty_cond = torch.zeros((batch_size, 77, 768)).to(self.device)
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            # UNet forward propagation
            self.unet(latents, timesteps, empty_cond)
        for key in self.features.keys():
            print(f"key: {key}, shape: {self.features[key].shape}")
        return self.features

def process_folders(folders, base_input_dir, base_output_dir, process_id):
    """Each process handles assigned folders"""
    device = torch.device(f'cuda:{process_id % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet"
    ).to(device)
    
    vae.eval()
    unet.eval()

    # Create feature extractor
    feature_extractor = FeatureExtractor(vae, unet, device)

    for folder in tqdm(folders, desc=f'Process {process_id}'):
        input_folder = os.path.join(base_input_dir, folder)
        output_folder = os.path.join(base_output_dir, folder)
        ensure_dir(output_folder)
        
        for file_name in os.listdir(input_folder):
            if not file_name.endswith('.png'):
                continue
                
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace('.png', '.npy'))
            
            try:
                features = feature_extractor.extract_features(input_path)
                np.save(output_path, features)
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                continue
            

def main():
    base_input_dir = "/home/ubuntu/scratch4/david/output/aligned_with_bg"
    base_output_dir = "/home/ubuntu/scratch4/david/output/sd_all_feat"
    ensure_dir(base_output_dir)

    # Get all HDTF folders
    folders = [d for d in os.listdir(base_input_dir) if d.startswith('HDTF_')]
    folders.sort()  # Ensure consistent order

    # Split folders into 4 chunks
    num_processes = 1
    folder_chunks = [folders[i::num_processes] for i in range(num_processes)]

    # Create processes
    processes = []
    for i in range(num_processes):
        p = Process(
            target=process_folders,
            args=(folder_chunks[i], base_input_dir, base_output_dir, i)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
