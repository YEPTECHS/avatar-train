import logging
import os
import cv2
import numpy as np

from concurrent.futures.thread import ThreadPoolExecutor
from face_alignment import FaceAlignment, LandmarksType
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from torch.utils.data import Dataset

os.makedirs("logs", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/training_{current_time}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Write to file
        # logging.StreamHandler(sys.stdout)  # Output to console
    ]
)

logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path

        video_path = Path(self.path)
        # self.video_files = list(video_path.glob('*/*/*.mp4'))
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
        self.fa = FaceAlignment(landmarks_type= LandmarksType.THREE_D, face_detector='sfd', device='cuda')


    def run(self):
        for i, video_path in enumerate(self.data):
            self._process_video(video_path, i)


    def _process_video(self, video_path:Path, completed: int):
        self.failed = 0
        self.logged = False
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.p_bar = tqdm(total=frame_count, desc=f'Processing {video_path.stem} [{completed}/{len(self.data)}] [0]')
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = self.detect_single_image(frame)

            image_name = '%06d' % i

            flag = ''
            lmk_dict = None
            # if no face detect change the image name
            if det is None:
                # no face detected
                flag = '[not_detect]'

            img_save_path = self.output_path / 'frame' /video_path.stem / (flag + image_name + '.png')
            landmark_save_path = self.output_path / 'landmarks' / video_path.stem / (flag + image_name + '.npz')
            # img_save_path = self.output_path / 'frame' /video_path.parent.stem / (flag + image_name + '.png')
            # landmark_save_path = self.output_path / 'landmarks' / video_path.parent.stem / (flag + image_name + '.npz')

            # submit to executor
            self.executor.submit(self._save_img, frame, img_save_path)

            # if no face detect change the image name
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
        # create the saving folder
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)

    def detect_single_image(self, img):
        bbox = self.fa.face_detector.detect_from_image(img)

        if len(bbox) == 0:
            return None

        else:
            if len(bbox) > 1:
                # if multiple boxes detected, use the one with highest confidence
                bbox = [bbox[np.argmax(np.array(bbox)[:, -1])]]

            lmks = self.fa.get_landmarks_from_image(img, detected_faces=bbox)[0]
        return bbox, lmks


class AudioSplit(object):
    def __init__(self, dataset, output_path):
        self.data = dataset
        self.output_path: Path = Path(output_path)

    def run(self):
        for video_path in tqdm(self.data, desc='Processing audio'):
            self._process_audio(video_path)

    def _process_audio(self, video_path:Path):
        # Read audio from video file
        audio = AudioSegment.from_file(str(video_path), format="mp4")
        audio_save_path = self.output_path / 'audio_ori' / video_path.stem / 'audio.mp3'
        audio_save_path.parent.mkdir(parents=True, exist_ok=True)
        # Export audio file
        audio.export(str(audio_save_path), format="mp3")



