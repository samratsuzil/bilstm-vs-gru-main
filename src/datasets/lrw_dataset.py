import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import random

class LRWDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, num_frames=25, augment=True):
        """
        data_dir: path to 'lrw'
        split: train / val / test
        transform: torchvision transforms
        num_frames: number of frames to sample per video
        augment: whether to apply data augmentation (only for training)
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.augment = augment and (split == "train")

        self.samples = []
        self.word2idx = {}
        idx = 0

        for word in sorted(os.listdir(data_dir)):
            word_path = os.path.join(data_dir, word, split)
            if not os.path.isdir(word_path):
                continue
            if word not in self.word2idx:
                self.word2idx[word] = idx
                idx += 1
            for file in sorted(os.listdir(word_path)):
                if file.endswith(".mp4"):
                    mp4_path = os.path.join(word_path, file)
                    self.samples.append((mp4_path, self.word2idx[word], word))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, word = self.samples[idx]
        frames = self.read_video(video_path)

        if self.augment:
            frames = self.apply_augmentation(frames)

        if self.transform:
            transformed_frames = []
            for frame in frames:
                frame_np = (frame.squeeze(0).numpy() * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np, mode='L')
                frame_transformed = self.transform(frame_pil)
                if not isinstance(frame_transformed, torch.Tensor):
                    import torchvision.transforms.functional as TF
                    frame_transformed = TF.to_tensor(frame_transformed)
                if frame_transformed.shape[0] == 3:
                    frame_transformed = frame_transformed.mean(dim=0, keepdim=True)
                transformed_frames.append(frame_transformed)
            frames = transformed_frames

        frames = torch.stack(frames)
        return frames, torch.tensor(label)

    def apply_augmentation(self, frames):
        """
        Apply aggressive data augmentation techniques for 85%+ accuracy
        - Random brightness adjustment (more aggressive)
        - Random contrast adjustment (more aggressive)
        - Horizontal flip
        - Random rotation
        - Gaussian noise
        - Random erasing (simulate occlusion)
        - Temporal speed perturbation (frame dropout)
        - Random Gaussian blur
        """
        augmented_frames = []

        # More aggressive augmentation parameters
        apply_flip = random.random() < 0.5
        brightness_factor = random.uniform(0.7, 1.3)  # More aggressive
        contrast_factor = random.uniform(0.7, 1.3)    # More aggressive
        rotation_angle = random.uniform(-10, 10)      # More aggressive
        add_noise = random.random() < 0.4             # More frequent
        apply_blur = random.random() < 0.3            # Gaussian blur
        apply_erasing = random.random() < 0.2         # Random erasing

        for frame in frames:
            frame_np = frame.squeeze(0).numpy()

            # Brightness adjustment
            frame_np = np.clip(frame_np * brightness_factor, 0, 1)

            # Contrast adjustment
            mean = frame_np.mean()
            frame_np = np.clip((frame_np - mean) * contrast_factor + mean, 0, 1)

            # Horizontal flip
            if apply_flip:
                frame_np = np.fliplr(frame_np)

            # Rotation
            if abs(rotation_angle) > 0.1:
                h, w = frame_np.shape
                M = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1.0)
                frame_np = cv2.warpAffine(frame_np, M, (w, h), borderValue=0)

            # Gaussian noise
            if add_noise:
                noise = np.random.normal(0, 0.03, frame_np.shape)  # Slightly more noise
                frame_np = np.clip(frame_np + noise, 0, 1)

            # Gaussian blur (simulate motion blur or out-of-focus)
            if apply_blur:
                kernel_size = random.choice([3, 5])
                frame_np = cv2.GaussianBlur(frame_np, (kernel_size, kernel_size), 0)

            # Random erasing (simulate occlusion)
            if apply_erasing:
                h, w = frame_np.shape
                erase_h = random.randint(10, 30)
                erase_w = random.randint(10, 30)
                erase_x = random.randint(0, w - erase_w)
                erase_y = random.randint(0, h - erase_h)
                frame_np[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = 0

            augmented_frames.append(torch.tensor(frame_np.copy(), dtype=torch.float32).unsqueeze(0))

        # Temporal augmentation: randomly drop some frames and repeat others
        if random.random() < 0.3 and len(augmented_frames) > 10:
            # Drop 1-2 random frames and duplicate others to maintain length
            num_drops = random.randint(1, 2)
            drop_indices = random.sample(range(len(augmented_frames)), num_drops)
            for idx in sorted(drop_indices, reverse=True):
                augmented_frames.pop(idx)
            # Duplicate random frames to maintain original length
            while len(augmented_frames) < len(frames):
                augmented_frames.insert(random.randint(0, len(augmented_frames)),
                                      augmented_frames[random.randint(0, len(augmented_frames)-1)])

        return augmented_frames

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (112, 112))
            frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0) / 255.0
            frames.append(frame)
        cap.release()

        if self.num_frames and len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif self.num_frames and len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        return frames
