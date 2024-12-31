import json
import logging
import time
import wandb
import math
import torch
import lpips
import os

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from typing import Dict, List
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from models.musetalk_dataset import MusetalkTrainDataset, MusetalkValDataset
from models.position_encoding import PositionalEncoding
from models.config import config
from models.loss import l1_loss, ssim
from models.repa import REPAHead
from helpers import split_dataset, frozen_params, preprocess_img_tensor


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
        self.weight_dtype_unet = self.weight_dtype_encoder = (
            torch.bfloat16 if config.train.mixed_precision == "bf16" else torch.float16
        )
        self.weight_dtype_decoder = torch.float32

        # Set model
        self.vae_decoder: AutoencoderKL = AutoencoderKL.from_pretrained(
            config.model.vae_pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype_decoder,
        )
        self.vae_encoder: AutoencoderKL = AutoencoderKL.from_pretrained(
            config.model.vae_pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype_encoder,
        )

        # Close dropout layers
        self.vae_encoder.eval()
        self.vae_decoder.eval()

        # Freeze the encoder and decoder
        frozen_params(self.vae_decoder)
        frozen_params(self.vae_encoder)

        # Unload the decoder and encoder to save memory
        self.vae_decoder.encoder = None
        self.vae_encoder.decoder = None
        logger.info("successfully loaded vae model")

        self.unet: UNet2DConditionModel = UNet2DConditionModel(**self.unet_config)
        logger.info("successfully loaded unet model")

        self.pe = PositionalEncoding(d_model=384).to(dtype=self.weight_dtype_unet)
        self.pe.eval()
        frozen_params(self.pe)
        logger.info("successfully loaded positional encoding")

        self.lpips_loss_fn = lpips.LPIPS(net="vgg").to(dtype=torch.float32)
        self.lpips_loss_fn.eval()
        frozen_params(self.lpips_loss_fn)

        if config.train.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.repa_head = REPAHead(
            in_channels_list=[320, 640, 1280],  # First 3 downblock channels
            hidden_dim=512,
            out_dim=384,
        ).to(dtype=torch.float32)

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
            list(self.unet.parameters()) + list(self.repa_head.parameters()),
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
        repa_features_gt: torch.Tensor = batch[5]
        ref_sd_features: torch.Tensor = batch[6]

        ref_imgs = preprocess_img_tensor(ref_imgs).to(self.accelerator.device)
        source_imgs = preprocess_img_tensor(source_imgs).to(self.accelerator.device)
        masked_source_imgs = preprocess_img_tensor(masked_source_imgs).to(
            self.accelerator.device
        )

        # Process audio features without moving to device
        audio_features = self.pe(audio_features.to(self.accelerator.device))
        repa_features_gt = repa_features_gt.to(self.accelerator.device)

        return ref_imgs, source_imgs, masked_source_imgs, _masks, audio_features, repa_features_gt, ref_sd_features

    def training_step(
        self,
        ref_imgs: torch.Tensor,
        source_imgs: torch.Tensor,
        repa_features_gt: torch.Tensor,
        masked_source_imgs: torch.Tensor,
        audio_features: torch.Tensor,
        ref_sd_features: Dict[str, List[np.ndarray]],
    ):
        logger.info(f"audio_features shape: {audio_features.shape}")
        # Encode source image and masked source image to latent space
        with torch.no_grad():
            latents: torch.Tensor = self.vae_encoder.encode(
                source_imgs.to(dtype=self.weight_dtype_encoder)
            ).latent_dist.sample() * self.vae_encoder.config.scaling_factor

            masked_latents: torch.Tensor = self.vae_encoder.encode(
                masked_source_imgs.reshape(source_imgs.shape).to(
                    dtype=self.weight_dtype_encoder
                )
            ).latent_dist.sample() * self.vae_encoder.config.scaling_factor

            ref_latents: torch.Tensor = self.vae_encoder.encode(
                ref_imgs.reshape(source_imgs.shape).to(dtype=self.weight_dtype_encoder)
            ).latent_dist.sample() * self.vae_encoder.config.scaling_factor

            # Set timesteps
            timesteps = torch.tensor([0], device=self.accelerator.device)
            latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

        conv_features = []
        def get_block_output(name):
            def hook(module, input, output):
                # Store only the output tensor, not the entire tuple
                conv_features.append(output[0] if isinstance(output, tuple) else output)
            return hook

        hooks = []
        # Only hook the output of first two downblocks
        for i in range(3):
            block = self.unet.down_blocks[i]
            hooks.append(block.register_forward_hook(get_block_output(f"down_block_{i}")))

        # Forward Pass
        latents_hat = self.unet(
            latent_model_input, timesteps, encoder_hidden_states=audio_features, 
            features_dict=ref_sd_features
        ).sample.float()

        proj_features = self.repa_head(conv_features[:3])
        for hook in hooks:
            hook.remove()

        # Decode latent vectors to image
        latents_hat = (
            latents_hat  / self.vae_encoder.config.scaling_factor
        )  # unscale vae latent
        image_hat = self.vae_decoder.decode(
            latents_hat.to(dtype=self.weight_dtype_decoder)
        ).sample 
        image_hat = image_hat[:, :, image_hat.shape[2] // 2 :, :]
        image_gt = source_imgs[:, :, source_imgs.shape[2] // 2 :, :]

        latents = latents.float()
        latents_hat = latents_hat.float()
        image_hat = image_hat.float()
        image_gt = image_gt.float()

        loss_lip = l1_loss(image_hat, image_gt)
        loss_latents = l1_loss(latents_hat * self.vae_encoder.config.scaling_factor, latents)
        D_SSIM_loss = 1.0 - ssim(image_hat, image_gt)
        lpips_loss = self.lpips_loss_fn(
            image_hat, image_gt, normalize=True
        ).mean()

        # Add debug prints
        logger.info(f"Initial proj_features shape: {proj_features.shape}")
        logger.info(f"Initial repa_features_gt shape: {repa_features_gt.shape}")
        
        # Normalize features
        proj_features = F.normalize(proj_features, dim=-1).float()
        repa_features_gt = F.normalize(repa_features_gt.reshape(repa_features_gt.shape[0], -1), dim=-1).float()
        
        logger.info(f"Final proj_features shape: {proj_features.shape}")
        logger.info(f"Final repa_features_gt shape: {repa_features_gt.shape}")

        # Calculate cosine similarity loss
        repa_loss = 1 - F.cosine_similarity(proj_features, repa_features_gt).mean()

        old_loss = (
            2.0 * loss_lip
            + loss_latents * 0.6
            + D_SSIM_loss * 0.4
            + lpips_loss * 1e-2
        )

        loss = old_loss + repa_loss * 0.5

        return loss_lip, loss_latents, D_SSIM_loss, lpips_loss, repa_loss, old_loss, loss

    @torch.no_grad()
    def validate(
        self,
        val_dataloader: DataLoader,
        epoch: int,
        global_step: int,
        dataset_mapping: Dict[int, int],
        dataset_metadata: dict,
        ref_sd_features: Dict[str, List[np.ndarray]],
    ):
        """
        Validation function that processes each sample individually and saves results by dataset
        """
        self.unet.eval()

        # Create base validation directory
        val_base_dir = os.path.join(config.data.output_dir, "validation_images", str(global_step))
        os.makedirs(val_base_dir, exist_ok=True)

        start = time.time()
        total_val_loss = 0
        num_val_samples = 0
        processed_samples = {}

        for step, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
            # Get original dataset index from mapping and create dataset directory
            dataset_idx = dataset_mapping[step]
            dataset_name = dataset_metadata[dataset_idx]["folder_name"]
            dataset_dir = os.path.join(val_base_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            if processed_samples.get(dataset_idx, 0) >= 50:
                continue

            # Process batch
            ref_imgs, source_imgs, masked_source_imgs, _masks, audio_features, repa_features_gt = (
                self.preprocess_batch(batch)
            )

            # Encode images to latent space
            latents = self.vae_encoder.encode(
                source_imgs.to(dtype=self.weight_dtype_encoder)
            ).latent_dist.sample()

            masked_latents = self.vae_encoder.encode(
                masked_source_imgs.to(dtype=self.weight_dtype_encoder)
            ).latent_dist.sample()

            ref_latents = self.vae_encoder.encode(
                ref_imgs.to(dtype=self.weight_dtype_encoder)
            ).latent_dist.sample()

            # Set timesteps and prepare input
            timesteps = torch.tensor([0], device=self.accelerator.device)
            latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

            # Forward pass through UNet
            conv_features = []
            def get_block_output(name):
                def hook(module, input, output):
                    conv_features.append(output[0] if isinstance(output, tuple) else output)
                return hook

            hooks = []
            # Only hook the output of first two downblocks
            for i in range(3):
                block = self.unet.down_blocks[i]
                hooks.append(block.register_forward_hook(get_block_output(f"down_block_{i}")))

            predicted_latents = self.unet(
                latent_model_input, timesteps, encoder_hidden_states=audio_features, 
                features_dict=ref_sd_features
            ).sample

            # Get REPA features
            proj_features = self.repa_head(conv_features[:3])
            for hook in hooks:
                hook.remove()

            # Calculate validation losses
            predicted_latents = predicted_latents / self.vae_encoder.config.scaling_factor
            predicted_image = self.vae_decoder.decode(
                predicted_latents.to(dtype=self.weight_dtype_decoder)
            ).sample

            # Calculate lower half losses
            predicted_image_lower = predicted_image[:, :, predicted_image.shape[2] // 2:, :]
            source_imgs_lower = source_imgs[:, :, source_imgs.shape[2] // 2:, :]

            val_loss_lip = l1_loss(predicted_image_lower.float(), source_imgs_lower.float())
            val_loss_latents = l1_loss(predicted_latents.float(), latents.float())
            val_D_SSIM_loss = 1.0 - ssim(predicted_image_lower.float(), source_imgs_lower.float())
            val_lpips_loss = self.lpips_loss_fn(
                predicted_image_lower.float(),
                source_imgs_lower.float(),
                normalize=True,
            ).mean()

            # Calculate REPA loss - updated to match training logic
            repa_features_gt = repa_features_gt.squeeze(1).reshape(repa_features_gt.shape[0], -1)
            proj_features_norm = F.normalize(proj_features, dim=-1)
            repa_features_gt_norm = F.normalize(repa_features_gt, dim=-1)
            val_repa_loss = 1 - F.cosine_similarity(proj_features_norm, repa_features_gt_norm).mean()

            # Calculate total loss matching training
            val_old_loss = (
                2.0 * val_loss_lip
                + val_loss_latents * 0.6
                + val_D_SSIM_loss * 0.4
                + val_lpips_loss * 1e-2
            )
            val_total_loss = val_old_loss + val_repa_loss

            total_val_loss += val_total_loss
            num_val_samples += 1

            # Create comparison image using original and predicted images
            comparison = Image.new("RGB", (config.data.IMAGE_SIZE * 4, config.data.IMAGE_SIZE))
            comparison.paste(self.postprocess_image(masked_source_imgs), (0, 0))
            comparison.paste(self.postprocess_image(ref_imgs), (config.data.IMAGE_SIZE, 0))
            comparison.paste(self.postprocess_image(source_imgs), (config.data.IMAGE_SIZE * 2, 0))
            comparison.paste(self.postprocess_image(predicted_image), (config.data.IMAGE_SIZE * 3, 0))

            # Save image with step number
            image_path = os.path.join(dataset_dir, f"val_step_{step}.png")
            comparison.save(image_path)

            # Log validation metrics for this sample
            if self.accelerator.is_main_process:
                wandb.log({
                    f"val/l1_loss": val_loss_lip.item(),
                    f"val/latents_loss": val_loss_latents.item(),
                    f"val/ssim_loss": val_D_SSIM_loss.item(),
                    f"val/lpips_loss": val_lpips_loss.item(),
                    f"val/repa_loss": val_repa_loss.item(),
                    f"val/old_loss": val_old_loss.item(),
                    f"val/total_loss": val_total_loss.item(),
                })
            processed_samples[dataset_idx] = processed_samples.get(dataset_idx, 0) + 1

        # Calculate and log average validation metrics
        avg_val_loss = total_val_loss / num_val_samples
        if self.accelerator.is_main_process:
            wandb.log({
                "val/average_loss": avg_val_loss,
                "val/epoch": epoch,
                "val/global_step": global_step,
                "val/time_taken": time.time() - start,
            })

        logger.info(f"Validation completed at step {global_step}. Average loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def postprocess_image(self, tensor: torch.Tensor, index=0):
        tensor = (tensor[index] + 1) / 2
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu().float()
        return transforms.ToPILImage(mode="RGB")(tensor)

    def _log_training_info(
        self, epoch: int, global_step: int, loss_lip, loss_latents, D_SSIM_loss, lpips_loss, repa_loss, old_loss, loss
    ):
        """记录训练信息到wandb和logger
        
        Args:
            epoch (int): 当前训练轮次
            global_step (int): 全局训练步数
            loss_lip (torch.Tensor): 唇部L1损失
            loss_latents (torch.Tensor): 潜空间L1损失  
            D_SSIM_loss (torch.Tensor): SSIM损失
            lpips_loss (torch.Tensor): LPIPS感知损失
            repa_loss (torch.Tensor): REPA损失
            old_loss (torch.Tensor): 旧损失
            loss (torch.Tensor): 总损失
            lr_scheduler: 学习率调度器
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
            f"(L1: {loss_lip.item():.4f}, "
            f"Latents: {loss_latents.item():.4f}, "
            f"SSIM: {D_SSIM_loss.item():.4f}, "
            f"LPIPS: {lpips_loss.item():.4f}), "
            f"Repa: {repa_loss.item():.4f}, "
            f"Old: {old_loss.item():.4f}, "
            f"LR: {lr:.6f}, "
            f"GPU: {gpu_memory:.2f}GB"
        )

        # Log to wandb
        wandb.log(
            {
                "train/total_loss": loss.item(),
                "train/l1_loss": loss_lip.item(),
                "train/l1_loss_latents": loss_latents.item(),
                "train/ssim_loss": D_SSIM_loss.item(),
                "train/lpips_loss": lpips_loss.item(),
                "train/repa_loss": repa_loss.item(),
                "train/old_loss": old_loss.item(),
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
                unwrapped_repa = self.accelerator.unwrap_model(self.repa_head)

                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "unet_state_dict": unwrapped_unet.state_dict(),
                        "repa_state_dict": unwrapped_repa.state_dict(),
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

    def train(self, validation_size: int):
        try:
            train_set = MusetalkTrainDataset(validation_size=validation_size)
            val_set = MusetalkValDataset(validation_size=validation_size)
            logger.info("Starting training")
            # Split dataset into train and validation sets
            '''train_set, val_set = split_dataset(
                dataset=dataset,
                train_ratio=config.train.dataset_split_ratio,
            )'''
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

            # Setup optimizer and lr scheduler
            self.setup_optimizer()
            logger.info("Optimizer and lr scheduler setup successfully")

            self.vae_encoder.to(self.accelerator.device)
            self.vae_decoder.to(self.accelerator.device)
            self.pe.to(self.accelerator.device)
            self.lpips_loss_fn.to(self.accelerator.device)
            logger.info("Model moved to device")
            
            # Use accelerator to wrap model, optimizer, and dataloaders
            (
                self.unet,
                self.repa_head,
                self.optimizer,
                train_dataloader,
                val_dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.unet,
                self.repa_head,
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

            global_step = 0
            actual_global_step = 0
            actual_accumulated_steps = 0

            # Start training
            for epoch in range(config.train.max_train_steps):
                self.unet.train()  # Train mode
                for _, batch in enumerate(train_dataloader):
                    logger.info(f"Epoch {epoch} Step {global_step}")

                    ref_imgs, source_imgs, masked_source_imgs, masks, audio_features, repa_features_gt, ref_sd_features = (
                        self.preprocess_batch(batch)
                    )
                    
                    logger.info("Batch Preprocessed")

                    with self.accelerator.accumulate(self.unet), self.accelerator.accumulate(self.repa_head):
                        actual_accumulated_steps += 1

                        # Train one step
                        loss_lip, loss_latents, D_SSIM_loss, lpips_loss, repa_loss, old_loss, loss = (
                            self.training_step(
                                ref_imgs,
                                source_imgs,
                                repa_features_gt,
                                masked_source_imgs,
                                audio_features,
                                ref_sd_features,
                            )
                        )
                        
                        logger.info("Training Step Completed")

                        # Backward pass
                        self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        logger.info(f"Syncing gradients: {actual_accumulated_steps}")
                        self.accelerator.clip_grad_norm_(
                            list(self.unet.parameters()) + list(self.repa_head.parameters()),
                            config.optimizer.max_grad_norm
                        )
                        actual_global_step += 1
                        progress_bar.update(1)
                        self._log_training_info(
                            epoch=epoch,
                            global_step=actual_global_step,
                            loss_lip=loss_lip,
                            loss_latents=loss_latents,
                            D_SSIM_loss=D_SSIM_loss,
                            lpips_loss=lpips_loss,
                            repa_loss=repa_loss,
                            old_loss=old_loss,
                            loss=loss,
                        )

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                # Move validation outside of gradient accumulation
                if actual_global_step > 0 and (actual_global_step % config.train.validation_steps) == 0:
                    if self.accelerator.is_main_process:
                        # Set model to eval mode for validation
                        self.unet.eval()
                        
                        val_loss = self.validate(
                            val_dataloader=val_dataloader,
                            epoch=epoch,
                            global_step=actual_global_step,
                            dataset_mapping=val_dataset_mapping,
                            dataset_metadata=val_dataset_metadata,
                            ref_sd_features=ref_sd_features,
                        )
                        
                        # Save checkpoint after validation
                        self.save_checkpoint(
                            epoch, actual_global_step, loss.item()
                        )
                        
                        # Set model back to training mode
                        self.unet.train()

                global_step += 1

                if actual_global_step >= config.train.max_train_steps:
                    logger.info(
                        f"Reached max_train_steps ({config.train.max_train_steps})"
                    )
                    return

            self.accelerator.wait_for_everyone()

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e

        finally:
            self.accelerator.end_training()