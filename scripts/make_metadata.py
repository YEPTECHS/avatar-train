import os
import json
import numpy as np
import argparse

def generate_metadata(image_root_dir: str, audio_root_dir: str, output_path: str):
    """Generate metadata.npz from directories of images and audio files
    
    Args:
        image_root_dir (str): Root directory containing HDTF-* folders with images
        audio_root_dir (str): Root directory containing HDTF-* folders with audio .npy files
        output_path (str): Path to save the metadata.npz file
    """
    folder_data = []
    
    for folder_name in sorted(os.listdir(image_root_dir)):
        image_folder_path = os.path.join(image_root_dir, folder_name)
        audio_folder_path = os.path.join(audio_root_dir, folder_name)
        
        if not os.path.isdir(image_folder_path):
            continue
            
        # Check if corresponding audio folder exists
        if not os.path.isdir(audio_folder_path):
            continue
            
        # Get all image names without extension and sort them
        image_names = []
        for img_file in os.listdir(image_folder_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                image_name = os.path.splitext(img_file)[0]
                # Check if corresponding .npy file exists
                npy_path = os.path.join(audio_folder_path, f"{image_name}.npy")
                if os.path.exists(npy_path):
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
    parser.add_argument('--image-dir', required=True,
                      help='Input directory path for images')
    parser.add_argument('--audio-dir', required=True,
                      help='Input directory path for audio .npy files')
    parser.add_argument('--output', default='metadata.npz',
                      help='Output path for metadata.npz (default: metadata.npz)')
    
    args = parser.parse_args()
    
    if args.mode == 'scan':
        generate_metadata(args.image_dir, args.audio_dir, args.output)
    else:  # json mode
        load_from_json(args.input, args.output)

if __name__ == '__main__':
    main()