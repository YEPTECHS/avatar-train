# THIS IS EXPERIMENTAL CODE
import concurrent.futures
from datetime import datetime
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from rich.console import Console

import logging
import os


class Logger:
    def __init__(self, log_file: str = None, log_level: int = logging.INFO, to_console: bool = True):
        """
        Initialize the logger.

        Parameters:
        - log_file (str): Path to the log file. If None, no file logging will be performed.
        - log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        - to_console (bool): Whether to log messages to the console.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Avoid duplicate handlers
        if not self.logger.hasHandlers():
            # File handler
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setLevel(log_level)
                file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)

            # Console handler
            if to_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)
                console_format = logging.Formatter('%(levelname)s - %(message)s')
                console_handler.setFormatter(console_format)
                self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Return the logger instance."""
        return self.logger


# Utility function to simplify logger creation
def get_logger(log_file: str = None, log_level: int = logging.INFO, to_console: bool = False) -> logging.Logger:
    """
    A simple utility to get a logger.

    Parameters:
    - log_file (str): Path to the log file.
    - log_level (int): Logging level.
    - to_console (bool): Whether to log messages to the console.

    Returns:
    - logging.Logger: Configured logger instance.
    """
    return Logger(log_file, log_level, to_console).get_logger()



class FaceAligner:
    def __init__(self, padding=0.10, output_size=(256, 256)):
        """
        Initialize FaceAligner class.

        Parameters:
        - padding: Horizontal padding ratio (default 0.10).
        - output_size: Output image size (width, height).
        """
        self.padding = padding
        self.output_size = output_size

    def rotate_image_and_landmarks(self, image, landmarks, angle, center):
        """
        Rotate image and landmarks.

        Parameters:
        - image: Input image.
        - landmarks: Landmarks array, shape (68, 2).
        - angle: Rotation angle (degrees).
        - center: Rotation center (x, y).

        Returns:
        - rotated_image: Rotated image.
        - rotated_landmarks: Rotated landmarks array.
        - M: Rotation matrix (2x3).
        """
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Rotate image
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        # Convert landmarks to homogeneous coordinates
        ones = np.ones((landmarks.shape[0], 1))
        landmarks_homogeneous = np.hstack([landmarks, ones])

        # Apply rotation matrix
        rotated_landmarks = M.dot(landmarks_homogeneous.T).T

        return rotated_image, rotated_landmarks, M

    def rotate_bbox(self, x1, y1, x2, y2, M):
        """
        Rotate bbox coordinates.

        Parameters:
        - x1, y1, x2, y2: Original bbox coordinates.
        - M: Rotation matrix (2x3).

        Returns:
        - x_rot_min, y_rot_min, x_rot_max, y_rot_max: Rotated bbox coordinates.
        """
        # Convert rotation matrix to 3x3
        M_full = np.vstack([M, [0, 0, 1]])

        # Define four corners
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])

        # Convert to homogeneous coordinates
        ones = np.ones((corners.shape[0], 1))
        corners_homogeneous = np.hstack([corners, ones])

        # Apply rotation matrix
        rotated_corners = M_full.dot(corners_homogeneous.T).T

        # Calculate new bbox
        x_rot_min = int(np.min(rotated_corners[:, 0]))
        y_rot_min = int(np.min(rotated_corners[:, 1]))
        x_rot_max = int(np.max(rotated_corners[:, 0]))
        y_rot_max = int(np.max(rotated_corners[:, 1]))

        return x_rot_min, y_rot_min, x_rot_max, y_rot_max

    def visualize_alignment(self, image, landmarks, rotated_image, rotated_landmarks, x1, y1, x2, y2):
        """
        Visualize the image and landmarks before and after alignment.

        Parameters:
        - image: Original image.
        - landmarks: Original landmarks.
        - rotated_image: Rotated image.
        - rotated_landmarks: Rotated landmarks.
        - x1, y1, x2, y2: Cropping area coordinates.
        """
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=10)

        # Rotated image
        plt.subplot(1, 2, 2)
        plt.title("Rotated Image with Landmarks and Crop")
        plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
        plt.scatter(rotated_landmarks[:, 0], rotated_landmarks[:, 1], c='r', s=10)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='g', facecolor='none', linewidth=2))

        plt.show()

    def align_and_crop_face(self, image, landmarks, bbox, visualize=False):
        """
        Align and crop face image based on landmarks and bbox.

        Parameters:
        - image: Input image.
        - landmarks: Landmarks array, shape (68, 2).
        - bbox: Original bbox coordinates [x1, y1, x2, y2].
        - visualize: Whether to visualize the alignment result.

        Returns:
        - cropped_aligned_face: Aligned and cropped image.
        """
        # Define indices for left and right eyes
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))

        # Calculate centers of left and right eyes
        left_eye_pts = landmarks[left_eye_indices]
        right_eye_pts = landmarks[right_eye_indices]
        left_eye_center = left_eye_pts.mean(axis=0)
        right_eye_center = right_eye_pts.mean(axis=0)

        # Calculate angle between eyes
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate rotation center (midpoint of eyes)
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                       (left_eye_center[1] + right_eye_center[1]) / 2.0)

        # Rotate image and landmarks
        rotated_image, rotated_landmarks, M = self.rotate_image_and_landmarks(image, landmarks, angle, eyes_center)

        # Get original bbox
        x1, y1, x2, y2 = bbox

        # Rotate bbox
        rotated_bbox = self.rotate_bbox(x1, y1, x2, y2, M)

        # Calculate size of rotated bbox
        x_rot_min, y_rot_min, x_rot_max, y_rot_max = rotated_bbox
        bbox_width = x_rot_max - x_rot_min
        bbox_height = y_rot_max - y_rot_min

        # Calculate padding
        pad_x = int(bbox_width * self.padding)
        pad_y = int(bbox_height * self.padding)

        # Apply padding, ensuring it doesn't exceed image boundaries
        x_rot_min_padded = max(x_rot_min - pad_x, 0)
        y_rot_min_padded = max(y_rot_min - pad_y, 0)
        x_rot_max_padded = min(x_rot_max + pad_x, rotated_image.shape[1])
        y_rot_max_padded = min(y_rot_max + pad_y, rotated_image.shape[0])

        # Crop image
        cropped_image = rotated_image[y_rot_min_padded:y_rot_max_padded, x_rot_min_padded:x_rot_max_padded]

        # Resize cropped image
        cropped_aligned_face = cv2.resize(cropped_image, self.output_size, interpolation=cv2.INTER_CUBIC)

        # Visualize alignment前后
        if visualize:
            self.visualize_alignment(image, landmarks, rotated_image, rotated_landmarks,
                                     x_rot_min_padded, y_rot_min_padded, x_rot_max_padded, y_rot_max_padded)

        return cropped_aligned_face

# Example usage
def process_image(image_path, lmk_path, face_aligner, logger , save_path, visualize=False):
    # Convert paths to Path objects
    image_path = Path(image_path)
    lmk_path = Path(lmk_path)
    save_path = Path(save_path)

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Can't read image: {image_path}")
        return

    # Read landmarks data
    try:
        lmk_data = np.load(str(lmk_path))
    except Exception as e:
        logger.error(f"Can't load landmarks file: {lmk_path}\nError: {e}, image_path: {image_path}")
        return

    # Check if landmarks data exists
    if 'face_landmark_2d' not in lmk_data:
        logger.error(f"Can't find 'face_landmark_2d' in landmarks file: {lmk_path}, image_path: {image_path}")
        return

    # Get landmarks
    landmarks = lmk_data['face_landmark_2d'][:, :2]
    bbox = lmk_data['bounding_box']
    bbox = bbox[0][:4]

    # Check landmarks count
    if landmarks.shape[0] != 68:
        logger.error(f"Expected 68 landmarks, but detected {landmarks.shape[0]} landmarks, image_path: {image_path}")
        return

    if len(bbox) != 4:
        logger.error(f"bbox should contain 4 values, but detected {len(bbox)} values, image_path: {image_path}")
        return

    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        logger.error(f"bbox values should be positive, but detected {bbox}, image_path: {image_path}")
        return
    # check if the bbox is out of the image
    if x1 >= image.shape[1] or y1 >= image.shape[0] or x2 >= image.shape[1] or y2 >= image.shape[0]:
        logger.error(f"bbox values out of image range, bbox: {bbox}, image shape: {image.shape} , image_path: {image_path}")
        return

    # Align and crop face
    cropped_aligned_face = face_aligner.align_and_crop_face(image, landmarks, bbox, visualize=visualize)

    # Save result
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cropped_aligned_face)


class CropImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
    ):
        """
        Custom image dataset

        Args:
            data_root (Path): Data root directory, containing 'frame' and 'landmarks' subdirectories
        """
        self.data_root = data_root

        self.frame_path = self.data_root / 'frame'
        self.lmk_path = self.data_root / 'landmarks'


        if not self.frame_path.exists():
            raise ValueError(f"Frame directory does not exist: {self.frame_path}")

        # Efficiently collect all image paths matching the naming format
        self.image_paths = self._collect_image_paths()

        if not self.image_paths:
            raise ValueError(f"No images found in {self.frame_path} matching pattern '000001.png'.")

    def _collect_image_paths(self) -> List[Path]:
        """
        Efficiently collect all image paths matching the naming format (six-digit number.png)

        Returns:
            list of Path: List of image paths matching the naming format
        """
        image_paths = []
        console = Console()

        # Get all subdirectories
        subdirs = [p for p in self.frame_path.iterdir() if p.is_dir()]
        # subdirs = [Path('/home/ubuntu/scratch4/david/output/mat_img/HDTF_0291')]
        total_subdirs = len(subdirs)

        if total_subdirs == 0:
            console.print(f"[red]No subdirectories found in {self.frame_path}.[/red]")
            return image_paths

        # Set progress bar style
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )

        def process_subdir(subdir: Path) -> List[Path]:
            """
            Process a single subdirectory, collecting image paths matching the naming format

            Args:
                subdir (Path): Subdirectory path

            Returns:
                List[Path]: List of image paths matching the naming format
            """
            local_paths = []
            try:
                for p in subdir.glob('*.png'):
                    stem = p.stem
                    if len(stem) == 6 and stem.isdigit():
                        local_paths.append(p)
            except Exception as e:
                console.print(f"[red]Error processing {subdir}: {e}[/red]")
            return local_paths

        with progress:
            task = progress.add_task("Loading image paths...", total=total_subdirs)
            # Use ThreadPoolExecutor to process subdirectories in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all subdirectory processing tasks
                futures = {executor.submit(process_subdir, subdir): subdir for subdir in subdirs}
                for future in concurrent.futures.as_completed(futures):
                    subdir = futures[future]
                    try:
                        local_paths = future.result()
                        image_paths.extend(local_paths)
                    except Exception as e:
                        console.print(f"[red]Error processing {subdir}: {e}[/red]")
                    finally:
                        progress.advance(task, 1)


        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}.")
        # idx = (idx + 1) * -1

        image_path = self.image_paths[idx]
        # load the landmarks
        video_name = image_path.parent.name
        frame_index = image_path.stem
        lmk_path = self.lmk_path / video_name / f"{frame_index}.npz"

        save_path = self.data_root / 'aligned_with_bg' / video_name / f"{frame_index}.png"

        return [str(image_path), str(lmk_path), str(save_path)]

# Main function example
if __name__ == "__main__":
    # Create FaceAligner instance
    face_aligner = FaceAligner(padding=0.1, output_size=(256, 256))

    # Configure path
    data_root = Path('/home/ubuntu/scratch4/david/output')  # Replace with your source directory path

    # Create dataset instance
    dataset = CropImageDataset(
        data_root=data_root,
    )


    # Process a single image
    # process_image(image_path, lmk_path, face_aligner, save_path='aligned_cropped.jpg', visualize=True)

    # Create a rich console
    console = Console()


    # Create a progress bar for processing
    progress = Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    # Set concurrency limit
    max_in_flight = 1000  # Adjust based on your CPU cores

    curent_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = get_logger(f'/home/ubuntu/scratch4/david/logs/training_{curent_time}.log')

    with progress:
        task = progress.add_task("Processing images...", total=len(dataset))
        # Use ThreadPoolExecutor for multi-thread processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            in_futures = set()
            for data in dataset:
                # data is a tuple: (image, image_path, lmk, bbox)
                image_path, lmk_path, save_path = data

                # Submit task
                future = executor.submit(process_image, image_path, lmk_path,face_aligner , logger, save_path, visualize=False)
                in_futures.add(future)
                # If maximum in-flight tasks are reached, wait for a task to complete
                if len(in_futures) >= max_in_flight:
                    done, in_futures = concurrent.futures.wait(in_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for completed_future in done:
                        try:
                            completed_future.result()
                        except Exception as e:
                            console.print(f"[red]Error processing image: {e}[/red]")
                        finally:
                            progress.advance(task, 1)
            # Process remaining futures
            for completed_future in concurrent.futures.as_completed(in_futures):
                try:
                    completed_future.result()
                except Exception as e:
                    console.print(f"[red]Error processing image: {e}[/red]")
                finally:
                    progress.advance(task, 1)

    console.print("[bold green]All images processed successfully.[/bold green]")

