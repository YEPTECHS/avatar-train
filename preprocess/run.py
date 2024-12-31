import os
import argparse

from .audio_feature_extractor import AudioFeatureExtractor
from .background_matter import BackgroundMatter
from .segment import VideoDataset, AudioSplit, Segment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video preprocessing pipeline')
    parser.add_argument('--data_path', type=str, 
                      default='/home/ubuntu/scratch4/david/tony',
                      help='Input data directory path')
    parser.add_argument('--output_path', type=str,
                      default='/home/ubuntu/scratch4/david/tony/output',
                      help='Output directory path')
    
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # start_index = 1
    # end_index = 292
    # base_url = "https://digital-human-llm.s3.ap-southeast-2.amazonaws.com/test/videos/HDTF/HDTF-"
    # asyncio.run(download_all_files(start_index, end_index, base_url))
    
    # Load dataset
    dataset = VideoDataset(data_path)

    # Split video into frames
    preprocesser = Segment(dataset, output_path)
    preprocesser.run()
    
    # Split audio
    audio_extracter = AudioSplit(dataset, output_path)
    audio_extracter.run()

    # Extract audio features
    audio_feature = AudioFeatureExtractor(dataset, output_path)
    audio_feature.run()

    # Used to remove background
    # background_matter = BackgroundMatter()
    # background_matter.run()