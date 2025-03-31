import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

class PathfinderDataset(Dataset):
    def __init__(self, root_dir, subdir='curv_baseline', transform=None):
        self.root = os.path.join(root_dir, subdir)
        self.img_dir = os.path.join(self.root, 'imgs')
        self.meta_dir = os.path.join(self.root, 'metadata')
        self.transform = transform

        self.samples = []
        for fname in sorted(os.listdir(self.meta_dir)):
            if not fname.endswith('.npy'):
                continue
            meta_path = os.path.join(self.meta_dir, fname)
            try:
                metadata = np.load(meta_path, allow_pickle=True)
            except Exception as e:
                with open(meta_path, 'r') as f:
                    lines = f.readlines()
                metadata = [line.strip().split() for line in lines if line.strip()]
                metadata = np.array(metadata)
            # [folder, filename, index, label, ...]
            for row in metadata:
                folder = row[0]
                file = row[1]
                label = int(row[3])

                img_path = os.path.join(self.root, folder, file)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')
        except (Image.UnidentifiedImageError, OSError) as e:
            print(f"[Skip bad img] {img_path}: {e}")
            dummy_np = np.zeros((32, 32), dtype=np.uint8)
            image = Image.fromarray(dummy_np, mode="L")
            label = 0
        if self.transform:
            image = self.transform(image)
        return image, label