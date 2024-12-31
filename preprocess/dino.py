import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from multiprocessing import Process
from pathlib import Path
from tqdm import tqdm

def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def process_folders(folders, src_base, dst_base, process_id):
    """Each process handles assigned folders"""
    # Initialize model (only initialized once per process)
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small')
    model.eval()
    device = torch.device('cuda:0')  # All processes use the same GPU
    model = model.to(device)

    # Process assigned folders
    for folder in tqdm(folders, desc=f'Process {process_id}'):
        src_folder = os.path.join(src_base, folder)
        dst_folder = os.path.join(dst_base, folder)
        ensure_dir(dst_folder)

        # Process all images in the folder
        for file in os.listdir(src_folder):
            if not file.endswith('.png'):
                continue

            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, f"{os.path.splitext(file)[0]}.npy")

            try:
                # Read and process image
                image = Image.open(src_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Extract features
                with torch.no_grad():
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    patch_features = outputs.last_hidden_state[:, 1:].cpu().numpy()

                # Save features
                np.save(dst_path, patch_features)

            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                continue

def main():
    src_base = "/home/ubuntu/scratch4/david/output/aligned"
    dst_base = "/home/ubuntu/scratch4/david/output/repa/dino"
    ensure_dir(dst_base)

    # Get all HDTF folders
    folders = [d for d in os.listdir(src_base) if d.startswith('HDTF_')]
    folders.sort()  # Ensure consistent order

    # Split folders into 4 chunks
    num_processes = 4
    folder_chunks = [folders[i::num_processes] for i in range(num_processes)]

    # Create processes
    processes = []
    for i in range(num_processes):
        p = Process(
            target=process_folders,
            args=(folder_chunks[i], src_base, dst_base, i)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()