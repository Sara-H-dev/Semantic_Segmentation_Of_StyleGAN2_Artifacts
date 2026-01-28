import cv2
import albumentations as A
from matplotlib import pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensorV2

# Beispielbild laden
img_path = "../dataset/fake_images/098000.png"  # <– hier dein Bildpfad
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define transformation pipeline (anpassbar)
transform = A.Compose([
    A.ToGray(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=8, p=0.8),
    A.OneOf([
        A.RandomGamma(gamma_limit=(90, 120), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.7),
])

# Mehrere Varianten erzeugen
num_samples = 6
augmented = [transform(image=image)["image"] for _ in range(num_samples)]

# Visualisierung
plt.figure(figsize=(16, 8))
plt.subplot(1, num_samples + 1, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

for i, img_aug in enumerate(augmented):
    plt.subplot(1, num_samples + 1, i + 2)
    plt.imshow(img_aug)
    plt.title(f"Aug {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()