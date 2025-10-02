import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# flips the image and label
def random_flip(image, label):
    image = np.flip(image, axis=1).copy()
    label = np.flip(label, axis=1).copy()
    return image, label



class RandomGenerator(object):
    def __init__(self, output_size, random_flip_flag = False):
        self.output_size = output_size
        self.random_flip_flag = random_flip_flag

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # PIL → NumPy
        image = np.array(image, dtype=np.float32)   # [H,W,3]
        label = np.array(label, dtype=np.uint8)    # [H,W]

        if self.random_flip_flag:
            if random.random() > 0.5:
                image, label = random_flip(image, label)

        H, W = image.shape[:2]
        
        if (H, W) != tuple(self.output_size):
            raise ValueError(f"Wrong image size: {H, W}")
        
        # Normalisation 0-1, if PNGs 0..255
        if image.max() > 1.0:
            image = image / 255.0
        label = (label > 127).astype(np.uint8)
        
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3,H,W]
        label = torch.from_numpy(label.astype(np.float32))

        return {'image': image, 'label': label}


# inherits from torch.utils.data.Dataset
class SegArtifact_dataset(Dataset):
    r"""
    SegArtifact_dataset

    Args:
        base_dir (str): root folder where the image and lable file lies
        list_dir (str): folder to file (which sample belongs to which split)
        split (str): options: "train", "val" and "test"
        transform: oprional pyTorch-Transfomration-pipeline (resize, flip, normalize)
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        # opens for example train.txt and reads all lines in the list
        with open(os.path.join(list_dir, self.split + '.txt'), 'r', encoding='utf-8') as f:
            self.sample_list = [ln.strip() for ln in f if ln.strip()]
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        # gets the name of the next data. 
        slice_name = self.sample_list[idx]
        # loads image and labels

        real_img_path = os.path.join(self.data_dir, "real_images", slice_name + ".png")
        fake_img_path = os.path.join(self.data_dir, "fake_images", slice_name + ".png")
        real_label_path = os.path.join(self.data_dir, "real_labels", slice_name + "_mask.png")
        fake_label_path = os.path.join(self.data_dir, "fake_labels", slice_name + "_mask.png")

        if os.path.exists(real_img_path):
            image = Image.open(real_img_path).convert("RGB")
            if os.path.exists(real_label_path):
                label = Image.open(real_label_path).convert("L")
            else: raise FileNotFoundError(f"Label {slice_name} not found in real_labels")

        elif os.path.exists(fake_img_path):
            image = Image.open(fake_img_path).convert("RGB")
            if os.path.exists(fake_label_path):
                label = Image.open(fake_label_path).convert("L")
            else: raise FileNotFoundError(f"Label {slice_name} not found in fake_labels")
        else:
            raise FileNotFoundError(f"Sample {slice_name} not found in real_images/ or fake_images/")

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
    
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
