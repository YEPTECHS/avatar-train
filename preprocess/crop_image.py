import cv2
import concurrent.futures
import numpy as np

from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset
from typing import List
from loguru import logger
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from rich.console import Console


class CropDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
    ):
        """
        Custom image dataset

        Args:
            data_root (Path): Root directory of the dataset, containing 'frame' and 'landmarks' subdirectories
        """
        self.data_root = data_root

        self.frame_path = self.data_root / 'frame'
        self.lmk_path = self.data_root / 'landmarks'


        if not self.frame_path.exists():
            raise ValueError(f"Frame directory does not exist: {self.frame_path}")

        # Collect all image paths with the naming format
        self.image_paths = self._collect_image_paths()

        if not self.image_paths:
            raise ValueError(f"No images found in {self.frame_path} matching pattern '000001.png'.")

    def _collect_image_paths(self) -> List[Path]:
        """
        Collect all image paths with the naming format (six-digit number.png)

        Returns:
            list of Path: List of image paths
        """
        image_paths = []
        console = Console()

        # Get all subdirectories
        subdirs = [p for p in self.frame_path.iterdir() if p.is_dir()]
        total_subdirs = len(subdirs)

        if total_subdirs == 0:
            console.print(f"[red]No subdirectories found in {self.frame_path}.[/red]")
            return image_paths

        # Set the style of the progress bar
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
            Process a single subdirectory, collect image paths that match the format

            Args:
                subdir (Path): Subdirectory path

            Returns:
                List[Path]: List of image paths
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

        image_path = self.image_paths[idx]
        # read the image using cv2
        image = cv2.imread(str(image_path))

        # load the landmarks
        video_name = image_path.parent.name
        frame_index = image_path.stem
        lmk_path = self.lmk_path / video_name / f"{frame_index}.npz"
        lmk_data = np.load(lmk_path)
        lmk = lmk_data['face_landmark_2d'][:, :2]
        bbox = lmk_data['bounding_box']

        return [image, str(image_path), lmk, bbox]


def crop_data(image, image_path, lmk, bbox, data_root) -> None:
    global logger
    # check if the image exist
    if image is None:
        logger.error(f"Image is None {str(image_path)}")
        return

    x1, y1, x2, y2, _ = bbox[0]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = max(0, int(x2))
    y2 = max(0, int(y2))

    y1 = y1 - 20
    y2 = y2 + 20
    x1 = x1 - 20
    x2 = x2 + 20

    if ((y2 - y1) <= 0) or ((x2 - x1) <= 0):
        logger.warning(f"Image Bbox out bound {str(image_path)}")
        return
    crop_frame = image[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    image_path = Path(image_path)
    video_name = image_path.parent.name
    frame_index = image_path.stem

    save_path = Path(data_root) / "aligned_with_bg" / str(video_name)
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / f"{frame_index}.png"
    cv2.imwrite(str(save_path), crop_frame)


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Configure path
    data_root = Path('/home/ubuntu/scratch4/david/tony/output')  # Replace with your source directory path

    # Create dataset instance
    dataset = CropDataset(
        data_root=data_root,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process each image independently
        shuffle=False,
        num_workers=8,  # Adjust based on CPU cores
        pin_memory=True,  # Enable if using GPU
        prefetch_factor=2,  # Improve prefetch efficiency
        persistent_workers=True  # Keep worker processes alive
    )

    # Define a function to process a single image tuple
    def process_image(image_tuple, data_root):
        image, image_path, lmk, bbox = image_tuple
        crop_data(image, image_path, lmk, bbox, data_root)

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
    max_in_flight = 1000  # Adjust based on needs

    curent_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # logger = get_logger(f'/home/ubuntu/scratch4/david/logs/training_{curent_time}.log"')

    with progress:
        task = progress.add_task("Processing images...", total=len(dataset))
        # Use ThreadPoolExecutor for multi-threaded processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            in_futures = set()
            for data in dataloader:
                # data is a tuple: (image, image_path, lmk, bbox)
                image, image_path, lmk, bbox = data
                # Unpack single sample
                image = image[0].numpy()  # Extract from batch
                image_path = str(image_path[0])  # Convert to string
                lmk = lmk[0].numpy()
                bbox = bbox[0].numpy()
                # Submit task
                future = executor.submit(process_image, (image, image_path, lmk, bbox), data_root)
                in_futures.add(future)
                # If maximum in-flight tasks reached, wait for one task to complete
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