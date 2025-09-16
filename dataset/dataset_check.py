import os
import random
import torch
from torch.utils.data import DataLoader

# --- importiere deine Klassen ---
from dataset import SegArtifact_dataset, RandomGenerator

# ------- Pfade anpassen -------
BASE_DIR = os.path.join("./")  # enthält images/ und masks/
LIST_DIR = os.path.join("../lists")    # enthält train.txt/val.txt/test.txt

# ------- Test-Parameter -------
SPLIT = "train"              # oder "val"/"test"
EXPECTED_SIZE = (1024, 1024)   # <== MUSS zu deinen PNGs passen (H, W)!
BATCH_SIZE = 2
NUM_SAMPLES_TO_SHOW = 3

def main():
    # Reproduzierbarkeit
    random.seed(0)
    torch.manual_seed(0)

    # Dataset aufsetzen
    transform = RandomGenerator(output_size=EXPECTED_SIZE)
    ds = SegArtifact_dataset(base_dir=BASE_DIR, list_dir=LIST_DIR, split=SPLIT, transform=transform)

    print(f"Split='{SPLIT}' | #Samples: {len(ds)}")
    if len(ds) == 0:
        print("Achtung: Die Listen-Datei ist leer. Bitte prüfe lists/train.txt.")
        return

    # Einzelne Samples holen
    for i in range(min(NUM_SAMPLES_TO_SHOW, len(ds))):
        sample = ds[i]
        img, lbl, name = sample["image"], sample["label"], sample["case_name"]
        # Shapes/Dtypes prüfen
        print(f"[{i}] {name}: image={tuple(img.shape)} {img.dtype}, label={tuple(lbl.shape)} {lbl.dtype}")
        # Erwartung: image=[3,H,W], label=[H,W]
        assert img.ndim == 3 and img.shape[0] == 3, "Image sollte [3,H,W] sein."
        assert lbl.ndim == 2, "Label sollte [H,W] sein."
        assert tuple(img.shape[1:]) == EXPECTED_SIZE and tuple(lbl.shape) == EXPECTED_SIZE, \
            f"Size-Mismatch. Erwartet {EXPECTED_SIZE}, bekam image {tuple(img.shape[1:])}, label {tuple(lbl.shape)}."
        # Labelwerte checken (binär 0/1 oder 0/255)
        unique_vals = torch.unique(lbl)
        print(f"   unique(label) = {unique_vals.tolist()}")
        if unique_vals.max() > 1:
            print("   Hinweis: Label sind nicht {0,1}. Falls sie {0,255} sind, wirst du im Loss evtl. vorher binarisieren wollen.")

    # DataLoader-Test (Batching)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    b_img, b_lbl = batch["image"], batch["label"]
    print(f"Batch image: {tuple(b_img.shape)} {b_img.dtype} | Batch label: {tuple(b_lbl.shape)} {b_lbl.dtype}")
    # Erwartung: [B,3,H,W] und [B,H,W]
    assert b_img.ndim == 4 and b_img.shape[1] == 3
    assert b_lbl.ndim == 3

    print("✅ Quick check passed.")

if __name__ == "__main__":
    main()
