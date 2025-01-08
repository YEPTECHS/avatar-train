import logging
import os
import cv2
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from torch.utils.data import Dataset
from face_alignment import FaceAlignment, LandmarksType


os.makedirs("logs", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/training_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
    ]
)

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        video_path = Path(self.path)
        self.video_files = sorted(list(video_path.glob('*.mp4')))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        return self.video_files[idx]

class Segment(object):
    def __init__(self, dataset, output_path):
        self.data = dataset
        self.executor = ThreadPoolExecutor()
        self.output_path: Path = Path(output_path)
        self.fa = FaceAlignment(
            LandmarksType.TWO_D, 
            face_detector='sfd',
            device='cuda'
        )

    def detect_single_image(self, img):
        h, w = img.shape[:2]
        min_size = min(h, w)
        if min_size < 64:
            scale = 64.0 / min_size
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        bbox = self.fa.face_detector.detect_from_image(img)
        
        if len(bbox) == 0:
            return None
        
        if len(bbox) > 1:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bbox]
            bbox = [bbox[np.argmax(areas)]]
            
        try:
            landmarks = self.fa.get_landmarks_from_image(img, detected_faces=bbox)[0]
            
            # Use all landmarks to determine the face area
            face_points = landmarks
            
            # Get the basic boundary
            x_min = np.min(face_points[:, 0])
            x_max = np.max(face_points[:, 0])
            y_min = np.min(face_points[:, 1])
            y_max = np.max(face_points[:, 1])
            
            # Calculate the center point and size
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            face_width = x_max - x_min
            face_height = y_max - y_min
            
            # Use a smaller expansion ratio
            
            # Leave a little more space for hair at the top
            top_margin = 0.2  # Expand 20% of the face height upwards
            bottom_margin = 0.1  # Expand 10% of the face height downwards
            side_margin = 0.1  # Expand 10% of the face width on each side
            
            # Calculate the final boundary box
            left_x = center_x - face_width * (0.5 + side_margin)
            right_x = center_x + face_width * (0.5 + side_margin)
            top_y = y_min - face_height * top_margin
            bottom_y = y_max + face_height * bottom_margin
            
            # Ensure it doesn't exceed the image boundaries
            left_x = max(0, int(left_x))
            top_y = max(0, int(top_y))
            right_x = min(w, int(right_x))
            bottom_y = min(h, int(bottom_y))
            
            # Final boundary box
            bbox = [[left_x, top_y, right_x, bottom_y, 1.0]]
            return bbox, landmarks
            
        except Exception as e:
            logger.warning(f"Error in landmark detection: {str(e)}")
            return None

    def run(self):
        for i, video_path in enumerate(self.data):
            self._process_video(video_path, i)

    def _process_video(self, video_path:Path, completed: int):
        self.failed = 0
        self.logged = False
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Add frame-to-frame smoothing
        prev_bbox = None
        prev_landmarks = None
        
        self.p_bar = tqdm(total=frame_count, desc=f'Processing {video_path.stem} [{completed}/{len(self.data)}] [0]')
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            det = self.detect_single_image(frame)
            
            # If the current frame fails detection but there is a previous frame result, use the previous frame result
            if det is None and prev_bbox is not None:
                det = (prev_bbox, prev_landmarks)
            elif det is not None:
                prev_bbox, prev_landmarks = det

            image_name = '%06d' % i

            flag = ''
            lmk_dict = None
            if det is None:
                flag = '[not_detect]'

            img_save_path = self.output_path / 'frame' /video_path.stem / (flag + image_name + '.png')
            landmark_save_path = self.output_path / 'landmarks' / video_path.stem / (flag + image_name + '.npz')

            self.executor.submit(self._save_img, frame, img_save_path)

            if det is None:
                self.p_bar.update(1)
                self.failed += 1
                if self.failed >= 100:
                    if not self.logged:
                        logger.warning(f"Failed to detect face in {video_path.stem}")
                        self.logged = True
                self.p_bar.set_description(
                    f'Processing {video_path.stem} [{completed}/{len(self.data)}] [{self.failed}]'
                )
                continue

            lmk_dict = {
                "bounding_box": det[0],
                "face_landmark_2d": det[1],
            }

            self.executor.submit(self._save_landmark, lmk_dict, landmark_save_path)
            self.p_bar.update(1)
        cap.release()

    def _save_landmark(self, lmk_dict, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **lmk_dict)

    def _save_img(self, frame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)

class AudioSplit(object):
    def __init__(self, dataset, output_path):
        self.data = dataset
        self.output_path: Path = Path(output_path)

    def run(self):
        for video_path in tqdm(self.data, desc='Processing audio'):
            self._process_audio(video_path)

    def _process_audio(self, video_path:Path):
        audio = AudioSegment.from_file(str(video_path), format="mp4")
        audio = audio.set_frame_rate(16000)
        audio_save_path = self.output_path / 'audio_ori' / video_path.stem / 'audio.mp3'
        audio_save_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(audio_save_path), format="mp3")



