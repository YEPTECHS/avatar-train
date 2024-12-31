import json
import logging
import time
import wandb
import math
import torch
import os

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from models.musetalk_dataset import MusetalkTrainDataset, MusetalkValDataset
from models.position_encoding import PositionalEncoding
from models.config import config
from lpips import LPIPS
from helpers import frozen_params, preprocess_img_tensor
from torchmetrics.image.fid import FrechetInceptionDistance


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        project_config: ProjectConfiguration,
    ):
        with open(config.model.unet_config_file, "r") as f:
            self.unet_config = json.load(f)

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            mixed_precision=config.train.mixed_precision,
            log_with="wandb" if config.train.wandb_enabled else None,
            project_config=project_config,
        )

        # Set mixed precision training
        self.weight_dtype = torch.float16

        # Set model - Use TAESD
        self.vae = AutoencoderTiny.from_pretrained(
            config.model.vae_pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype
        )
        
        # Close dropout layers
        self.vae.eval()
        # Freeze the vae
        frozen_params(self.vae)
        logger.info("successfully loaded taesd model")

        self.unet: UNet2DConditionModel = UNet2DConditionModel(**self.unet_config)
        logger.info("successfully loaded unet model")

        self.pe = PositionalEncoding(d_model=384).to(dtype=self.weight_dtype)
        self.pe.eval()
        frozen_params(self.pe)
        logger.info("successfully loaded positional encoding")

        self.loss_module = LossModule()
        logger.info("successfully initialized loss module")

        if config.train.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def setup_optimizer(self):
        """Initialize optimizer with settings parameters"""
        # Scale learning rate if specified
        if config.optimizer.scale_lr:
            learning_rate = (
                config.optimizer.learning_rate
                * config.train.gradient_accumulation_steps
                * config.train.train_batch_size
            )
        else:
            learning_rate = config.optimizer.learning_rate

        optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            list(self.unet.parameters()),
            lr=learning_rate,
            betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
            weight_decay=config.optimizer.adam_weight_decay,
            eps=config.optimizer.adam_epsilon,
        )

        lr_scheduler = get_scheduler(
            config.optimizer.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.optimizer.lr_warmup_steps
            * self.accelerator.num_processes,
            num_training_steps=config.train.max_train_steps
            * self.accelerator.num_processes,
        )

        self.optimizer: torch.optim.Optimizer = optimizer
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = lr_scheduler

    def create_dataloader(self, dataset, is_train=True, world_size=None, rank=None):
        """
        Create a DataLoader from a dataset.

        Args:
            dataset: Dataset object
            config: Config object
            is_train: Is training or not
            world_size: Current world size
            rank: Current rank
        """
        batch_size = config.train.train_batch_size if is_train else 1

        if world_size is not None and rank is not None:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train,
                drop_last=is_train,
            )
        else:
            sampler = None

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(
                sampler is None and is_train
            ),  # If using sampler, shuffle must be False
            num_workers=24,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=True,
            prefetch_factor=8,
            sampler=sampler,
        )

        # TODO sampler need to be fixed
        return dataloader #, sampler

    def preprocess_batch(self, batch: tuple):
        ref_imgs: torch.Tensor = batch[0]
        source_imgs: torch.Tensor = batch[1]
        masked_source_imgs: torch.Tensor = batch[2]
        _masks: torch.Tensor = batch[3]  # not used
        audio_features: torch.Tensor = batch[4]

        ref_imgs = preprocess_img_tensor(ref_imgs).to(self.accelerator.device)
        source_imgs = preprocess_img_tensor(source_imgs).to(self.accelerator.device)
        masked_source_imgs = preprocess_img_tensor(masked_source_imgs).to(
            self.accelerator.device
        )

        # Process audio features without moving to device
        audio_features = self.pe(audio_features.to(self.accelerator.device))

        return ref_imgs, source_imgs, masked_source_imgs, _masks, audio_features

    def training_step(
        self,
        ref_imgs: torch.Tensor,
        source_imgs: torch.Tensor,
        masked_source_imgs: torch.Tensor,
        audio_features: torch.Tensor,
    ):
        # Encode source image and masked source image to latent space
        with torch.no_grad():
            latents: torch.Tensor = self.vae.encoder(
                source_imgs.to(dtype=self.weight_dtype)
            )

            masked_latents: torch.Tensor = self.vae.encoder(
                masked_source_imgs.to(dtype=self.weight_dtype)
            )

            ref_latents: torch.Tensor = self.vae.encoder(
                ref_imgs.to(dtype=self.weight_dtype)
            )

            # Set timesteps
            timesteps = torch.tensor([0], device=self.accelerator.device)
            latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

        # Forward Pass
        predicted_latents = self.unet(
            latent_model_input, timesteps, encoder_hidden_states=audio_features,
        ).sample

        # Decode latent vectors to image
        predicted_image = self.vae.decoder(predicted_latents.to(dtype=self.weight_dtype))

        # Make sure loss module is in training mode
        self.loss_module.train()
        # Calculate losses
        losses = self.loss_module(
            generated=predicted_image,
            target=source_imgs,
            identity_ref=ref_imgs
        )

        return (
            losses['l1_lower'],
            losses['lpips_loss'],
            losses['triplet_loss'],
            losses['total_loss']
        )

    @torch.no_grad()
    def validate(
        self,
        val_dataloader: DataLoader,
        epoch: int,
        global_step: int,
        dataset_mapping: Dict[int, int],
        dataset_metadata: dict,
    ):
        """Validation function that processes each sample individually and saves results by dataset"""
        self.unet.eval()
        # Set loss module to evaluation mode
        self.loss_module.eval()

        # Initialize FID calculator
        fid = FrechetInceptionDistance(normalize=True).to(self.accelerator.device)
        
        # Create base validation directory
        val_base_dir = os.path.join(config.data.output_dir, "validation_images", str(global_step))
        os.makedirs(val_base_dir, exist_ok=True)

        start = time.time()
        processed_samples = {}

        total_l1_lower = 0
        total_lpips_loss = 0
        total_triplet_loss = 0
        total_loss = 0
        num_samples = 0
        
        dataset_images = {}

        for step, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
            # Get original dataset index from mapping
            dataset_idx = dataset_mapping[step]
            
            # Skip if this dataset has already processed 50 images
            if processed_samples.get(dataset_idx, 0) >= 50:
                continue
            
            dataset_name = dataset_metadata[dataset_idx]["folder_name"]
            dataset_dir = os.path.join(val_base_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Process batch
            ref_imgs, source_imgs, masked_source_imgs, _masks, audio_features = (
                self.preprocess_batch(batch)
            )

            # Encode images to latent space using TAESD
            latents: torch.Tensor = self.vae.encoder(
                source_imgs.to(dtype=self.weight_dtype)
            )

            masked_latents: torch.Tensor = self.vae.encoder(
                masked_source_imgs.to(dtype=self.weight_dtype)
            )

            ref_latents: torch.Tensor = self.vae.encoder(
                ref_imgs.to(dtype=self.weight_dtype)
            )

            # Set timesteps and prepare input
            timesteps = torch.tensor([0], device=self.accelerator.device)
            latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

            predicted_latents = self.unet(
                latent_model_input, timesteps, encoder_hidden_states=audio_features,
            ).sample

            # Decode latent vectors to image using TAESD
            predicted_image = self.vae.decoder(predicted_latents.to(dtype=self.weight_dtype))

            # Calculate validation losses using forward pass
            val_losses = self.loss_module(
                generated=predicted_image,
                target=source_imgs,
                identity_ref=ref_imgs
            )

            # Save first 50 samples for each dataset
            processed_idx = processed_samples.get(dataset_idx, 0)
            if processed_idx < 50:
                comparison = Image.new("RGB", (config.data.IMAGE_SIZE * 4, config.data.IMAGE_SIZE))
                comparison.paste(self.postprocess_image(masked_source_imgs), (0, 0))
                comparison.paste(self.postprocess_image(ref_imgs), (config.data.IMAGE_SIZE, 0))
                comparison.paste(self.postprocess_image(source_imgs), (config.data.IMAGE_SIZE * 2, 0))
                comparison.paste(self.postprocess_image(predicted_image), (config.data.IMAGE_SIZE * 3, 0))

                # Save image with step number
                image_path = os.path.join(dataset_dir, f"val_{dataset_idx}_{processed_idx}.png")
                comparison.save(image_path)
                
                if dataset_idx not in dataset_images:
                    dataset_images[dataset_idx] = []
                dataset_images[dataset_idx].append(image_path)

                # Log validation metrics for this sample
                total_l1_lower += val_losses['l1_lower'].item()
                total_lpips_loss += val_losses['lpips_loss'].item()
                total_triplet_loss += val_losses['triplet_loss'].item()
                total_loss += val_losses['total_loss'].item()
                num_samples += 1

            processed_samples[dataset_idx] = processed_samples.get(dataset_idx, 0) + 1

            # Add real and generated images to FID calculator
            real_imgs_fid = (source_imgs * 255).to(torch.uint8)
            fake_imgs_fid = (predicted_image * 255).to(torch.uint8)
            
            # Update FID
            fid.update(real_imgs_fid, real=True)
            fid.update(fake_imgs_fid, real=False)

        # Only compute FID if we have enough samples
        try:
            fid_score = float(fid.compute())
        except RuntimeError:
            logger.warning("Not enough samples to compute FID score")
            fid_score = float('nan')
        
        if self.accelerator.is_main_process and num_samples > 0:
            wandb.log({
                f"val/l1_lower": total_l1_lower / num_samples,
                f"val/lpips_loss": total_lpips_loss / num_samples,
                f"val/triplet_loss": total_triplet_loss / num_samples,
                f"val/total_loss": total_loss / num_samples,
                f"val/fid_score": fid_score if not math.isnan(fid_score) else None,
            })

        # Create video for each dataset after validation loop
        if self.accelerator.is_main_process:
            for dataset_idx, image_paths in dataset_images.items():
                if len(image_paths) > 0:
                    dataset_name = dataset_metadata[dataset_idx]["folder_name"]
                    video_path = os.path.join(val_base_dir, f"{dataset_name}_validation.mp4")
                    self.create_video_from_images(image_paths, video_path)

        # Clean up FID calculator
        fid.reset()

        # Restore loss module to training mode (if needed)
        if self.unet.training:
            self.loss_module.train()

        logger.info(f"Validation time taken: {time.time() - start}")
        return total_loss / num_samples if num_samples > 0 else 0

    def create_video_from_images(self, image_paths, output_path, fps=24):
        """
        Create a video from a list of images using multiple codec fallbacks
        """
        try:
            import cv2
            import numpy as np
            
            # Read the first image to get dimensions
            first_image = cv2.imread(image_paths[0])
            if first_image is None:
                logger.error(f"Failed to read first image: {image_paths[0]}")
                return
            
            height, width, layers = first_image.shape

            # Try different codecs in order of preference
            codecs = [
                ('avc1', '.mp4'),  # H.264 codec
                ('H264', '.mp4'),  # Alternative H.264 name
                ('mp4v', '.mp4'),  # MPEG-4
            ]

            success = False
            for codec, ext in codecs:
                try:
                    # Update output path with correct extension
                    current_output = output_path.rsplit('.', 1)[0] + ext
                    
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(
                        current_output,
                        fourcc,
                        fps,
                        (width, height),
                        isColor=True
                    )

                    if not out.isOpened():
                        logger.warning(f"Failed to initialize VideoWriter with codec {codec}")
                        continue

                    # Add each image to video
                    for image_path in image_paths:
                        frame = cv2.imread(image_path)
                        if frame is not None:
                            out.write(frame)
                        else:
                            logger.warning(f"Failed to read image: {image_path}")

                    out.release()
                    
                    # Verify the video was created successfully
                    if os.path.exists(current_output) and os.path.getsize(current_output) > 0:
                        logger.info(f"Successfully created video with codec {codec} at {current_output}")
                        success = True
                        break
                    else:
                        logger.warning(f"Video file empty or not created with codec {codec}")
                        
                except Exception as e:
                    logger.warning(f"Failed to create video with codec {codec}: {str(e)}")
                    if out:
                        out.release()
                    continue

            if not success:
                logger.error("Failed to create video with any codec, saving images individually")
                # Fall back to saving individual images in a directory
                save_dir = output_path.rsplit('.', 1)[0] + '_frames'
                os.makedirs(save_dir, exist_ok=True)
                for idx, image_path in enumerate(image_paths):
                    frame = cv2.imread(image_path)
                    if frame is not None:
                        save_path = os.path.join(save_dir, f"frame_{idx:04d}.png")
                        cv2.imwrite(save_path, frame)
                logger.info(f"Saved individual frames to {save_dir}")

        except Exception as e:
            logger.error(f"Video creation failed with error: {str(e)}")
            raise

    def postprocess_image(self, tensor: torch.Tensor, index=0):
        tensor = (tensor[index] + 1) / 2
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu().float()
        return transforms.ToPILImage(mode="RGB")(tensor)

    def _log_training_info(
        self, epoch: int, global_step: int, loss_lip, lpips_loss, triplet_loss, loss
    ):
        """Record training information to wandb and standard logger
        
        Args:
            epoch (int): Current training epoch
            global_step (int): Global training step
            loss_lip (torch.Tensor): Lower face L1 loss
            lpips_loss (torch.Tensor): LPIPS perceptual loss
            triplet_loss (torch.Tensor): Triplet loss
            loss (torch.Tensor): Total loss
        """
        # Only log on main process
        if not self.accelerator.is_main_process:
            return
            
        # Get current learning rate and GPU memory usage
        lr = self.lr_scheduler.get_last_lr()[0]
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9
        
        # Log to standard logger
        logger.info(
            f"[Epoch {epoch} Step {global_step}] "
            f"Loss: {loss.item():.4f} "
            f"(L1 Lower: {loss_lip.item():.4f}, "
            f"LPIPS: {lpips_loss.item():.4f}, "
            f"Triplet: {triplet_loss.item():.4f}), "
            f"LR: {lr:.6f}, "
            f"GPU: {gpu_memory:.2f}GB"
        )

        # Log to wandb
        wandb.log(
            {
                "train/total_loss": loss.item(),
                "train/l1_lower_loss": loss_lip.item(),
                "train/lpips_loss": lpips_loss.item(),
                "train/triplet_loss": triplet_loss.item(),
                "train/learning_rate": lr,
                "train/gpu_memory_gb": gpu_memory,
                "train/epoch": epoch,
                "train/global_step": global_step,
            }
        )

    def save_checkpoint(self, epoch: int, step: int, loss):
        """Save checkpoint"""
        try:
            if self.accelerator.is_main_process:
                checkpoint_dir = os.path.join(config.data.output_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint-epoch{epoch}-step{step}.pth"
                )
                logger.info(f"Saving checkpoint to {checkpoint_path}")

                unwrapped_unet = self.accelerator.unwrap_model(self.unet)

                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "unet_state_dict": unwrapped_unet.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "loss": loss,
                    },
                    checkpoint_path
                )
                logger.info(f"Saved model checkpoint to {checkpoint_path}")

                accelerator_dir = os.path.join(checkpoint_dir, f"accelerator-epoch{epoch}-step{step}")
                self.accelerator.save_state(accelerator_dir)
                logger.info(f"Saved accelerator state to {accelerator_dir}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise e

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from .bin, .pth file or accelerate folder
        
        Args:
            checkpoint_path (str): Path to either:
                - .bin file containing model weights
                - .pth file containing full checkpoint state
                - accelerate checkpoint folder
                
        Returns:
            tuple: (start_epoch, global_step) for resuming training
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Default return values
        start_epoch = 0
        global_step = 0
        
        try:
            # Ensure model and optimizer are prepared before loading checkpoint
            if not hasattr(self, 'optimizer') or self.optimizer is None:
                self.setup_optimizer()
                logger.info("Initialized optimizer before loading checkpoint")
                
            if checkpoint_path.endswith('.bin'):
                # Load .bin format (just model weights)
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                unwrapped_unet = self.accelerator.unwrap_model(self.unet)
                # Add strict parameter checking
                missing_keys, unexpected_keys = unwrapped_unet.load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    logger.warning(f"Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
                logger.info("Successfully loaded .bin checkpoint (model weights only)")
                
            elif checkpoint_path.endswith('.pth'):
                # Load .pth format (full checkpoint state)
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Load model weights with strict checking
                unwrapped_unet = self.accelerator.unwrap_model(self.unet)
                missing_keys, unexpected_keys = unwrapped_unet.load_state_dict(
                    checkpoint["unet_state_dict"], strict=False
                )
                if missing_keys:
                    logger.warning(f"Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
                
                # Load optimizer state if available
                if "optimizer_state_dict" in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        logger.info("Successfully loaded optimizer state")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")
                    
                # Load scheduler state if available    
                if "scheduler_state_dict" in checkpoint:
                    try:
                        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        logger.info("Successfully loaded scheduler state")
                    except Exception as e:
                        logger.warning(f"Failed to load scheduler state: {str(e)}")
                    
                # Get training state
                start_epoch = checkpoint.get("epoch", 0)
                global_step = checkpoint.get("step", 0)
                
                logger.info(f"Successfully loaded .pth checkpoint - Epoch: {start_epoch}, Step: {global_step}")
                
            elif os.path.isdir(checkpoint_path):
                # Ensure all components are prepared before loading accelerator state
                self.unet, self.optimizer, _, _, self.lr_scheduler = self.accelerator.prepare(
                    self.unet, self.optimizer, None, None, self.lr_scheduler
                )
                
                # Load accelerator state
                self.accelerator.load_state(checkpoint_path)
                logger.info(f"Successfully loaded accelerator state from {checkpoint_path}")
                
                # Try to extract epoch and step from folder name
                folder_name = os.path.basename(checkpoint_path)
                if folder_name.startswith("accelerator-epoch"):
                    try:
                        parts = folder_name.split("-")
                        start_epoch = int(parts[1].replace("epoch", ""))
                        global_step = int(parts[2].replace("step", ""))
                        logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
                    except:
                        logger.warning("Could not parse epoch and step from folder name")
                
                # Validate model state
                self.unet.eval()
                with torch.no_grad():
                    try:
                        # Validate model state
                        test_input = torch.randn(1, self.unet.config.in_channels, 32, 32).to(self.accelerator.device)
                        test_timestep = torch.tensor([0]).to(self.accelerator.device)
                        test_encoder_hidden_states = torch.randn(1, 1, self.unet.config.cross_attention_dim).to(self.accelerator.device)
                        _ = self.unet(test_input, test_timestep, encoder_hidden_states=test_encoder_hidden_states)
                        logger.info("Model validation successful after loading checkpoint")
                    except Exception as e:
                        logger.error(f"Model validation failed after loading checkpoint: {str(e)}")
                        raise
                        
            else:
                raise ValueError(
                    f"Unsupported checkpoint format: {checkpoint_path}. "
                    "Must be either a .bin file, .pth file or an accelerate checkpoint folder."
                )
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise e
            
        # Ensure all components are on the correct device
        self.unet.to(self.accelerator.device)
        logger.info(f"Model moved to device: {self.accelerator.device}")
        
        # Validate optimizer state
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                logger.info(f"Current learning rate: {param_group['lr']}")
        
        return start_epoch, global_step

    def train(self, validation_size: int, resume_from_checkpoint: str = None):
        try:
            # Initialize starting epoch and step
            start_epoch = 0
            actual_global_step = 0
            
            # Setup optimizer and lr scheduler
            self.setup_optimizer()
            logger.info("Optimizer and lr scheduler setup successfully")
            
            # Load checkpoint if specified
            if resume_from_checkpoint:
                start_epoch, actual_global_step = self.load_checkpoint(resume_from_checkpoint)
                logger.info(f"Resuming training from epoch {start_epoch}, step {actual_global_step}")
            
            train_set = MusetalkTrainDataset(validation_size=validation_size)
            val_set = MusetalkValDataset(validation_size=validation_size)
            logger.info("Starting training")
            logger.info(
                f"Train set size: {len(train_set)}, Val set size: {len(val_set)}"
            )

            # Create dataloaders
            train_dataloader = self.create_dataloader(train_set, is_train=True)
            val_dataloader = self.create_dataloader(val_set, is_train=False)
            val_dataset_mapping: Dict[int, int] = val_set.dataset_mapping
            val_dataset_metadata: np.ndarray = val_set.dataset_metadata
            
            logger.info("Dataloader Created Successfully")

            total_samples = len(train_set)  # 910000 * 0.8 = 782000
            samples_per_update = (
                config.train.train_batch_size * config.train.gradient_accumulation_steps
            )  # 12 * 512 = 6400
            updates_per_epoch = math.ceil(
                total_samples / samples_per_update
            )  # 237 / 6400 = 1/36
            total_updates = (
                updates_per_epoch * config.train.max_train_steps
            )  # 1000000 / 36 = 27333

            logger.info("Training configuration:")
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Batch size: {config.train.train_batch_size}")
            logger.info(
                f"Gradient accumulation steps: {config.train.gradient_accumulation_steps}"
            )
            logger.info(f"Effective batch size: {samples_per_update}")
            logger.info(f"Updates per epoch: {updates_per_epoch}")
            logger.info(f"Total updates planned: {total_updates}")
            logger.info(
                f"Train set size: {len(train_set)}, Val set size: {len(val_set)}"
            )

            self.vae.to(self.accelerator.device)
            self.pe.to(self.accelerator.device)
            self.loss_module.to(self.accelerator.device)
            logger.info("Model moved to device")
            
            # Use accelerator to wrap model, optimizer, and dataloaders
            (
                self.unet,
                self.optimizer,
                train_dataloader,
                val_dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.unet,
                self.optimizer,
                train_dataloader,
                val_dataloader,
                self.lr_scheduler,
            )
            
            logger.info("Model, optimizer, and dataloaders prepared for training")

            if self.accelerator.is_main_process:
                self.accelerator.init_trackers("musetalk training")
                logger.info("Accelerator trackers initialized")

            progress_bar = tqdm(
                range(config.train.max_train_steps),
                disable=not self.accelerator.is_local_main_process,
            )
            progress_bar.set_description("Steps")

            actual_accumulated_steps = 0

            # Start training
            for epoch in range(config.train.max_train_steps):
                self.unet.train()
                for _, batch in enumerate(train_dataloader):
                    logger.info(f"Epoch {epoch} Step {actual_global_step}")

                    ref_imgs, source_imgs, masked_source_imgs, masks, audio_features = (
                        self.preprocess_batch(batch)
                    )
                    
                    logger.info("Batch Preprocessed")

                    with self.accelerator.accumulate(self.unet):
                        actual_accumulated_steps += 1

                        # Train one step
                        l1_lower, lpips_loss, triplet_loss, loss = (
                            self.training_step(
                                ref_imgs,
                                source_imgs,
                                masked_source_imgs,
                                audio_features,
                            )
                        )
                        
                        logger.info("Training Step Completed")

                        # Backward pass
                        self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        logger.info(f"Syncing gradients: {actual_accumulated_steps}")
                        self.accelerator.clip_grad_norm_(
                            list(self.unet.parameters()),
                            config.optimizer.max_grad_norm
                        )
                        actual_global_step += 1
                        progress_bar.update(1)
                        self._log_training_info(
                            epoch=epoch,
                            global_step=actual_global_step,
                            loss_lip=l1_lower,
                            lpips_loss=lpips_loss,
                            triplet_loss=triplet_loss,
                            loss=loss,
                        )

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                if actual_global_step > 0 and (actual_global_step % config.train.validation_steps) == 0:
                    self.run_validation(
                        val_dataloader=val_dataloader,
                        epoch=epoch,
                        global_step=actual_global_step,
                        dataset_mapping=val_dataset_mapping,
                        dataset_metadata=val_dataset_metadata,
                    )
                    
                    # Save checkpoint
                    self.save_checkpoint(epoch, actual_global_step, loss.item())

                if actual_global_step >= config.train.max_train_steps:
                    break

            self.accelerator.wait_for_everyone()

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e

        finally:
            self.accelerator.end_training()

    def run_validation(self, val_dataloader, epoch, global_step, dataset_mapping, dataset_metadata):
        """Separate validation function"""
        if not self.accelerator.is_main_process:
            return
        
        # Save current model state
        training_mode = self.unet.training
        
        # Set to evaluation mode
        self.unet.eval()
        
        val_loss = self.validate(
            val_dataloader=val_dataloader,
            epoch=epoch,
            global_step=global_step,
            dataset_mapping=dataset_mapping,
            dataset_metadata=dataset_metadata,
        )
        
        # Restore previous model state
        if training_mode:
            self.unet.train()
        
        return val_loss


class LossModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LPIPS(net='vgg').cuda()
        self.lower_face_ratio = 0.5
        
    def get_lower_face_mask(self, h, w):
        """Generate lower face mask"""
        mask = torch.zeros((h, w))
        mask[int(h * self.lower_face_ratio):, :] = 1
        return mask.cuda()

    def forward(self, generated, target, identity_ref):
        """
        Forward propagation function - calculate all losses
        """
        return self.compute_loss(generated, target, identity_ref)

    def compute_loss(self, generated, target, identity_ref):
        # Make sure the input is in float32
        generated = generated.float()
        target = target.float()
        identity_ref = identity_ref.float()

        b, c, h, w = generated.shape
        
        # 1. Lower face L1 loss
        lower_mask = self.get_lower_face_mask(h, w)
        lower_mask = lower_mask.expand(b, c, h, w)
        l1_lower = torch.mean(torch.abs(generated * lower_mask - target * lower_mask))
        
        # 2. LPIPS perceptual loss
        lpips_loss = self.lpips(generated, target).mean()
        
        # 3. Adaptive Triplet Loss
        with torch.no_grad():
            d_gt_ref = self.lpips(target, identity_ref).mean()
        d_gen_ref = self.lpips(generated, identity_ref).mean()
        d_gen_gt = self.lpips(generated, target).mean()
        
        triplet_loss = torch.clamp(d_gen_gt - (d_gen_ref / (d_gt_ref + 1e-7)) + 1, min=0)
        
        # Total loss
        total_loss = 1.0 * l1_lower + 2.0 * lpips_loss + 0.5 * triplet_loss
        
        return {
            'total_loss': total_loss,
            'l1_lower': l1_lower,
            'lpips_loss': lpips_loss,
            'triplet_loss': triplet_loss
        }
