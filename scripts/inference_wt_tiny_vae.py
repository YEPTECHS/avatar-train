import torch
import cv2
import subprocess
import shutil
import os
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from models.position_encoding import PositionalEncoding
from torchmetrics.image.fid import FrechetInceptionDistance
from facenet_pytorch import MTCNN, InceptionResnetV1



class VideoUNetInference:
    def __init__(
        self,
        unet_path: str,
        vae_path: str = "madebyollin/taesd",
        device: str = "cuda"
    ):
        self.device = device
        self.dtype = torch.float16
        
        # Load VAE - use single AutoencoderTiny instance
        self.vae = AutoencoderTiny.from_pretrained(
            vae_path,
            torch_dtype=self.dtype
        ).to(device)
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            unet_path,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=self.dtype
        ).to(device)

        self.pe = PositionalEncoding(d_model=384).to(dtype=self.dtype, device=device)
        self.pe.eval()
        
        self.vae.eval()
        self.unet.eval()

        # Init face detection and recognition model
        self.mtcnn = MTCNN(
            image_size=256,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def preprocess_img_tensor(image_tensor: torch.Tensor):
        # Assume input is a PyTorch tensor of shape (N, C, H, W)
        N, C, H, W = image_tensor.shape
        # Calculate new width and height, ensuring they are multiples of 32
        new_w = W - W % 32
        new_h = H - H % 32
        # Use torchvision.transforms method for scaling and resampling
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Apply transformation to each image and store result in a new tensor
        preprocessed_images = torch.empty((N, C, new_h, new_w), dtype=torch.float32)
        for i in range(N):
            # Use F.interpolate instead of transforms.Resize
            resized_image: torch.Tensor = F.interpolate(image_tensor[i].unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            preprocessed_images[i] = transform(resized_image.squeeze(0))

        return preprocessed_images

    def _reshape_image(self, img):
        #  H x W x 3
        x = np.expand_dims(img, axis=0)
        x = np.asarray(x) / 255.0
        x = np.transpose(x, (0,3,1,2))  # Convert to C x H x W
        return x

    @torch.no_grad()
    def preprocess_frame(self, frame: np.ndarray, ref_frame: np.ndarray = None) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        source_img = self._reshape_image(frame_rgb)

        # If ref_frame is provided, use it as reference, otherwise use source frame
        if ref_frame is not None:
            ref_frame_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            ref_img = self._reshape_image(ref_frame_rgb)
        else:
            ref_img = source_img.copy()

        # Mask the right half of source image
        source_img[:, :, source_img.shape[2] // 2:, :] = 0

        masked_source_img = torch.FloatTensor(source_img)
        masked_source_img = self.preprocess_img_tensor(masked_source_img)
        ref_img = torch.FloatTensor(ref_img)
        ref_img = self.preprocess_img_tensor(ref_img)

        return ref_img, source_img, masked_source_img

    @torch.no_grad()
    def process_frame(
        self,
        frame: np.ndarray,
        audio_feature: torch.Tensor = None,
        ref_frame: np.ndarray = None
    ) -> torch.Tensor:
        ref_img, source_img, masked_source_img = self.preprocess_frame(frame, ref_frame)

        # Move input tensors to correct device
        ref_img = ref_img.to(device=self.device)
        masked_source_img = masked_source_img.to(device=self.device)

        # Encode to latent space using single VAE - modified to match trainer
        ref_latent = self.vae.encoder(ref_img.to(dtype=self.dtype))
        masked_latent = self.vae.encoder(masked_source_img.to(dtype=self.dtype))
        
        # Prepare UNet input
        latent_input = torch.cat([masked_latent, ref_latent], dim=1)
        timesteps = torch.tensor([0], device=self.device, dtype=self.dtype)
        
        if audio_feature is None:
            audio_feature = torch.zeros(1, 50, 384, device=self.device, dtype=self.dtype) + 1e-6
            audio_feature = self.pe(audio_feature)
        
        # Run UNet forward pass
        output = self.unet(
            latent_input,
            timesteps,
            encoder_hidden_states=audio_feature
        ).sample
        
        # Decode output using same VAE - modified to match trainer
        decoded = self.vae.decoder(output.to(torch.float16))
        
        # Post-process output
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = (decoded * 255).round().to(torch.uint8)
        return decoded

    def process_image_sequence(
        self,
        input_dir: str,
        output_path: str,
        audio_features_dir: str = None,
        start_frame: int = 0,
        end_frame: int = None,
        fps: float = 25.0,
        temp_dir: str = "temp_frames",
        audio_length_left: int = 2,
        audio_length_right: int = 2
    ):
        """Process a sequence of PNG images from a directory"""
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        ref_temp_dir = temp_dir / "ref"
        ref_temp_dir.mkdir(parents=True, exist_ok=True)

        real_images = []
        generated_images = []
        
        try:
            # Get list of PNG files
            input_dir = Path(input_dir)
            png_files = sorted(list(input_dir.glob("*.png")))
            
            if not png_files:
                raise ValueError(f"No PNG files found in {input_dir}")
            
            # Get total number of frames
            total_frames = len(png_files)
            if end_frame is None:
                end_frame = total_frames
            else:
                end_frame = min(end_frame, total_frames)
            
            # Process each frame and save as image
            pbar = tqdm(total=end_frame - start_frame, desc="Processing frames")
            
            for frame_idx in range(start_frame, end_frame):
                # Read current frame
                frame = cv2.imread(str(png_files[frame_idx]))
                
                if frame is None:
                    continue
                
                # Get reference frame (furthest frame)
                ref_idx = end_frame - 1 if frame_idx < (end_frame + start_frame) // 2 else start_frame
                ref_frame = cv2.imread(str(png_files[ref_idx]))
                
                # Add original frame to real_images list
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                real_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
                real_images.append(real_tensor)
                
                # Load audio features with context
                if audio_features_dir is None:
                    # If no audio features are provided, use zero tensor
                    audio_feature = torch.zeros(1, 50, 384, device=self.device, dtype=self.dtype)
                    audio_feature = self.pe(audio_feature)
                else:
                    # Original audio features loading logic
                    audio_features = []
                    feature_shape = None
                    
                    # Load surrounding frames' features
                    for offset in range(-audio_length_left, audio_length_right + 1):
                        new_idx = frame_idx + offset
                        feature_path = os.path.join(audio_features_dir, f"{new_idx:06d}.npy")
                        
                        try:
                            feat = np.load(feature_path)
                            if feature_shape is None:
                                feature_shape = feat.shape
                            audio_features.append(feat)
                        except Exception as e:
                            # If feature not found, use zeros
                            if feature_shape is not None:
                                audio_features.append(None)
                            else:
                                continue
                    
                    for idx, audio_feature in enumerate(audio_features):
                        if audio_feature is None:
                            audio_features[idx] = np.zeros(feature_shape)

                    # Concatenate features
                    audio_feature = np.concatenate(audio_features, axis=0)
                    audio_feature = audio_feature.reshape(-1, 384)  # 384 是 whisper feature dimension
                    
                    # Convert to tensor and add positional encoding
                    audio_feature = torch.from_numpy(audio_feature).unsqueeze(0).to(
                        device=self.device,
                        dtype=self.dtype
                    )
                    audio_feature = self.pe(audio_feature)
                
                # Process frame with reference frame
                processed_tensor = self.process_frame(frame, audio_feature, ref_frame)
                processed_tensor = processed_tensor[0]
                generated_images.append(processed_tensor)
                
                # Convert to image when saving
                frame_path = temp_dir / f"frame_{frame_idx:06d}.png"
                save_tensor = processed_tensor.permute(1, 2, 0).cpu().numpy()
                Image.fromarray(save_tensor).save(str(frame_path))

                ref_frame_path = ref_temp_dir / f"frame_{frame_idx:06d}.png"
                Image.fromarray(frame_rgb).save(str(ref_frame_path))

                pbar.update(1)
            
            pbar.close()

            if real_images and generated_images:
                # Squeeze extra dimensions after stacking
                real_batch = torch.stack(real_images, dim=0).squeeze(1)
                generated_batch = torch.stack(generated_images, dim=0).squeeze(1)
                
                print(f"Real images shape: {real_batch.shape}, dtype: {real_batch.dtype}, range: [{real_batch.min()}, {real_batch.max()}]")
                print(f"Generated images shape: {generated_batch.shape}, dtype: {generated_batch.dtype}, range: [{generated_batch.min()}, {generated_batch.max()}]")
                
                fid_score = self.calculate_fid(real_batch, generated_batch)
                print(f"\nFID Score: {fid_score:.4f}")
                
                # Calculate CSIM
                csim_score = self.calculate_csim(real_batch, generated_batch)
                print(f"CSIM Score: {csim_score:.4f}")
            
            # Create output paths for two videos
            output_path = Path(output_path)
            ref_output_path = output_path.parent / f"ref_{output_path.name}"

            # Create processed video using ffmpeg
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(fps),
                '-i', str(temp_dir / 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                str(output_path)
            ]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

            # Create reference video using ffmpeg
            ffmpeg_cmd_ref = [
                'ffmpeg',
                '-y',
                '-framerate', str(fps),
                '-i', str(ref_temp_dir / 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                str(ref_output_path)
            ]
            subprocess.run(ffmpeg_cmd_ref, check=True, capture_output=True)
                
        except Exception as e:
            raise
            
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def preprocess_img_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Image preprocessing function"""
        N, C, H, W = image_tensor.shape
        # Ensure dimensions are multiples of 32
        new_w = W - W % 32
        new_h = H - H % 32
        
        # Normalization transformation
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        preprocessed_images = torch.empty((N, C, new_h, new_w), dtype=torch.float32)
        
        for i in range(N):
            # Resampling
            resized_image = F.interpolate(
                image_tensor[i].unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            # Normalization
            preprocessed_images[i] = transform(resized_image.squeeze(0))
        
        return preprocessed_images

    def calculate_fid(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate Fréchet Inception Distance between real and generated images.
        
        Args:
            real_images: Tensor of shape (N, C, H, W) in range [0, 255] (uint8)
            generated_images: Tensor of shape (N, C, H, W) in range [0, 255] (uint8)
            
        Returns:
            float: The calculated FID score
        """
        fid = FrechetInceptionDistance(normalize=True).to(self.device)
        
        fid.update(real_images.to(self.device, dtype=torch.uint8), real=True)
        fid.update(generated_images.to(self.device, dtype=torch.uint8), real=False)
        
        score = float(fid.compute())
    
        # Reset after calculation
        fid.reset()
        
        return score

    def calculate_csim(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate CSIM using BlazeFace detection and FaceNet embeddings.
        """
        # Initialize BlazeFace detector
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.01
        )
        
        similarities = []
        
        for idx, (real_img, gen_img) in enumerate(zip(real_images, generated_images)):
            try:
                # Convert to numpy array and ensure correct format
                real_np = real_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                gen_np = gen_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                # Detect faces
                real_results = face_detector.process(real_np)
                gen_results = face_detector.process(gen_np)
                
                if (not real_results.detections) or (not gen_results.detections):
                    print(f"No face detected in pair {idx}")
                    # Save problematic images for debugging
                    if idx < 5:
                        Image.fromarray(real_np).save(f'debug_real_{idx}.png')
                        Image.fromarray(gen_np).save(f'debug_gen_{idx}.png')
                    continue
                
                # Get face bounding boxes
                def get_bbox(detection, img_height, img_width):
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * img_width)
                    y = int(bbox.ymin * img_height)
                    w = int(bbox.width * img_width)
                    h = int(bbox.height * img_height)
                    return max(0, x), max(0, y), min(img_width, x + w), min(img_height, y + h)
                
                h, w = real_np.shape[:2]
                real_bbox = get_bbox(real_results.detections[0], h, w)
                gen_bbox = get_bbox(gen_results.detections[0], h, w)
                
                # Crop face regions
                real_face_img = real_np[real_bbox[1]:real_bbox[3], real_bbox[0]:real_bbox[2]]
                gen_face_img = gen_np[gen_bbox[1]:gen_bbox[3], gen_bbox[0]:gen_bbox[2]]
                
                # Resize to FaceNet required size
                real_face_img = cv2.resize(real_face_img, (160, 160))
                gen_face_img = cv2.resize(gen_face_img, (160, 160))
                
                # Convert to PyTorch tensor
                real_face_tensor = torch.from_numpy(real_face_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                gen_face_tensor = torch.from_numpy(gen_face_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                
                # Normalize
                real_face_tensor = (real_face_tensor - 127.5) / 128.0
                gen_face_tensor = (gen_face_tensor - 127.5) / 128.0
                
                # Get feature vectors
                with torch.no_grad():
                    real_embedding = self.facenet(real_face_tensor)
                    gen_embedding = self.facenet(gen_face_tensor)
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(real_embedding, gen_embedding).item()
                similarities.append(similarity)
                
                print(f"Successfully processed pair {idx} with similarity: {similarity:.4f}")
                
            except Exception as e:
                print(f"Error processing image pair {idx}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # Clean up resources
        face_detector.close()
        
        # Return average similarity
        if similarities:
            avg_sim = float(np.mean(similarities))
            print(f"\nProcessed {len(similarities)} valid face pairs out of {len(real_images)} total pairs")
            print(f"Similarity range: [{min(similarities):.4f}, {max(similarities):.4f}]")
            return avg_sim
        else:
            print("\nNo valid face pairs found!")
            return 0.0


# Usage example:
if __name__ == "__main__":
    model = VideoUNetInference(
        unet_path="/home/ubuntu/scratch4/kuizong/avatar-train/pretrained",
        vae_path="madebyollin/taesd"
    )
    
    # Process video
    model.process_image_sequence(
        input_dir="/home/ubuntu/scratch4/kuizong/avatar-train/dataset/processed/aligned_with_bg/tony",
        output_path="/home/ubuntu/scratch4/kuizong/avatar-train/output_finetune_tiny_new_loss_tony.mp4",
        #audio_features_dir="/home/ubuntu/scratch4/kuizong/avatar-train/dataset/processed/audio/tony_val",
        start_frame=0,
        end_frame=600
    )
