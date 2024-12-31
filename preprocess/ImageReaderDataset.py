import concurrent.futures

from pathlib import Path
from typing import List
from PIL import Image
from torchvision import transforms
from rich.console import Console
from torch.utils.data import Dataset
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn


class ImageReaderDataset(Dataset):
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

        self.transform = transforms.ToTensor()

        if not self.frame_path.exists():
            raise ValueError(f"Frame directory does not exist: {self.frame_path}")

        # Collect all image paths with the naming format
        self.image_paths = self._collect_image_paths()

        if not self.image_paths:
            raise ValueError(f"No images found in {self.frame_path} matching pattern '000001.png'.")

    def _collect_image_paths(self) -> List[Path]:
        """
        Efficiently collect all image paths with the naming format (six-digit number.png)

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


        with Image.open(str(self.image_paths[idx])) as img:
            img.load()
        return self.transform(img), self.image_paths[idx].parent.stem, self.image_paths[idx].stem


