import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------------------------------
# Einstellungen
# -------------------------------
fake_label_dir = "../fake_labels"        # Ordner mit Masken (0–1 oder 0–255)
base_image_path = "ground_picture.png"  # Hintergrundbild
output_path = "artifact_heatmap.png"    # Ergebnisbild
normalize_masks = True

# -------------------------------
# Heatmap berechnen
# -------------------------------
def create_heatmap(fake_label_dir, base_image_path, output_path):
    # Masken laden
    mask_files = [f for f in os.listdir(fake_label_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not mask_files:
        raise ValueError(f"Keine Masken gefunden im Ordner: {fake_label_dir}")

    # Erste Maske für Referenzgröße
    first_mask = np.asarray(Image.open(os.path.join(fake_label_dir, mask_files[0])).convert("L"))
    h, w = first_mask.shape

    # Akkumulator
    heatmap_accum = np.zeros((h, w), dtype=np.float32)

    print(f"length mask: {len(mask_files)}")

    # Masken summieren
    for fname in mask_files:
        mask = np.asarray(Image.open(os.path.join(fake_label_dir, fname)).convert("L").resize((w, h)))
        if normalize_masks:
            mask = mask / 255.0
        heatmap_accum += mask

    # Normalisieren auf [0,1]
    heatmap_norm = heatmap_accum / len(mask_files)

    # Basisbild laden
    base_img = Image.open(base_image_path).convert("RGB").resize((w, h))
    base_img = np.asarray(base_img) / 255.0  # auf [0,1]
    # -------------------------------
    # Ajust colors
    # -------------------------------
    cmap = plt.get_cmap("jet")
    colors = cmap(np.linspace(0.2, 1, 256))  # Start bei 0.1 statt 0.0 -> weniger dunkles Blau
    bright_jet = mcolors.LinearSegmentedColormap.from_list("bright_jet", colors)

    # -------------------------------
    # Heatmap rendern
    # -------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(base_img)
    plt.imshow(heatmap_norm, cmap=bright_jet, alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"✅ Heatmap gespeichert unter: {output_path}")

# -------------------------------
# Ausführung
# -------------------------------
if __name__ == "__main__":
    create_heatmap(fake_label_dir, base_image_path, output_path)
