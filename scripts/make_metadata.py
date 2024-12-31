import os
import json
import numpy as np
from pathlib import Path
import argparse

def generate_metadata(image_root_dir: str, output_path: str):
    """Generate metadata.npz from a directory of images
    
    Args:
        image_root_dir (str): Root directory containing HDTF-* folders with images
        output_path (str): Path to save the metadata.npz file
    """
    folder_data = []
    
    # Walk through all HDTF-* directories
    for folder_name in sorted(os.listdir(image_root_dir)):
        if not folder_name.startswith('HDTF'):
            continue
            
        folder_path = os.path.join(image_root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Get all image names without extension and sort them
        image_names = []
        for img_file in os.listdir(folder_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                image_name = os.path.splitext(img_file)[0]
                image_names.append(image_name)
        
        image_names.sort()
        
        if image_names:
            folder_info = {
                'folder_name': folder_name,
                'image_count': len(image_names),
                'image_names': image_names
            }
            folder_data.append(folder_info)
    
    # Save as npz
    if folder_data:
        metadata = {'folder_data': folder_data}
        np.savez(output_path, **metadata)
        print(f"Generated metadata.npz with {len(folder_data)} folders at {output_path}")
        
        # Also save as JSON for easy viewing
        json_path = output_path.replace('.npz', '.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Also saved as JSON at {json_path}")
    else:
        print("No valid folders found!")

def load_from_json(json_path: str, output_path: str):
    """Generate metadata.npz from a JSON file
    
    Args:
        json_path (str): Path to the input JSON file
        output_path (str): Path to save the metadata.npz file
    """
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    np.savez(output_path, **metadata)
    print(f"Generated metadata.npz from JSON at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate metadata.npz file')
    parser.add_argument('--mode', choices=['scan', 'json'], required=True,
                      help='Mode: scan directory or convert from JSON')
    parser.add_argument('--input', required=True,
                      help='Input directory path (for scan mode) or JSON file path (for json mode)')
    parser.add_argument('--output', default='metadata.npz',
                      help='Output path for metadata.npz (default: metadata.npz)')
    
    args = parser.parse_args()
    
    if args.mode == 'scan':
        generate_metadata(args.input, args.output)
    else:  # json mode
        load_from_json(args.input, args.output)

if __name__ == '__main__':
    main()