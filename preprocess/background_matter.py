import torch

from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from .ImageReaderDataset import ImageReaderDataset


class ImageDataset(Dataset):
    def __init__(self, dataroot):
        self.data = dataroot

        self.images = list(self.data.glob('*.png'))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with Image.open(str(self.images[idx])) as img:
            img.load()
        return self.transform(img), self.images[idx].parent.stem ,self.images[idx].stem


class BackgroundMatter:
    def __init__(self):
        self.output_path: Path = Path('./output')
        self.data = self.output_path / 'frame'
        # glob all the folders in the data directory
        self.data = list(self.data.glob('*'))
        self.data = [x for x in self.data if x.is_dir()]
        # sort the folders
        self.data.sort()
        # self.data = self.data[53:]

        # dataset = ImageReaderDataset(self.data)
        # print("dataset length", len(dataset))

        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").to('cuda')
        # self.convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

        self.executor = ThreadPoolExecutor()

        pass

    def run(self):
        for video in self.data:
            self._mat(video)

    def _mat(self, video):
        dataset = ImageDataset(video)
        self.data_loader = DataLoader(dataset, batch_size=64, num_workers=30, shuffle=False, pin_memory=True)
        with torch.no_grad():
            rec = [None] * 4
            for (batch, video_name, path) in tqdm(self.data_loader, desc=f"Processing {video.stem}", total=len(self.data_loader)):
                batch = batch.cuda().unsqueeze(0)
                downsample_ratio = self.auto_downsample_ratio(*batch.shape[3:])
                fgr, pha, *rec = self.model(batch, *rec, downsample_ratio)


                for fgr_img, pha_img, v, p in zip(fgr.squeeze(0), pha.squeeze(0), video_name, path):
                    output_path = self.output_path / 'mat_img' / v
                    alpha_path = self.output_path / 'alpha' / v

                    self.executor.submit(self._save_mat_image, fgr_img, pha_img, output_path / f'{p}.png')
                    self.executor.submit(self._save_img, pha_img, alpha_path / f'{p}.png')



    def _save_img(self, img, path):
        # check if path exist
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        to_pil_image(img).save(path)

    def _save_mat_image(self, fgr, pha, path):
        fgr = fgr * pha.gt(0)
        com = torch.cat([fgr, pha], dim=-3)
        self._save_img(com, path)


    def auto_downsample_ratio(self, h, w):
        """
        Automatically find a downsample ratio so that the largest side of the resolution be 512px.
        """
        return min(512 / max(h, w), 1)
        # foreground_path = self.output_path / 'foreground' / video.stem

        # self.convert_video(
        #     self.model,  # Model
        #     input_source=str(video),  # Video file or image sequence folder
        #     output_type='png_sequence',  # Optional "video" (video) or "png_sequence" (PNG sequence)
        #     output_composition=str(output_path),  # If exporting video, provide file path. If exporting PNG sequence, provide folder path
        #     output_alpha=str(alpha_path),  # [Optional] Output transparency prediction
        #     # output_foreground=str(foreground_path),  # [Optional] Output foreground prediction
        #     seq_chunk=12,  # Set multi-frame parallel calculation
        #     num_workers=1,  # Only applicable to image sequence input, read thread
        #     progress=True,  # Display progress bar
        #     device = device,
        #     dtype = torch.float32
        # )


