#!/usr/bin/env python3
import os
import random
from glob import glob

# ========= EINSTELLUNGEN =========
IMAGE_DIR = "../dataset/images"
MASK_DIR  = "../dataset/labels"   # erwartet: <basename>_mask.png
LISTS_DIR = "./"

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SHUFFLE = True
SEED = 42
LIMIT = None
# =================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def main():
    os.makedirs(LISTS_DIR, exist_ok=True)

    # Bilder einsammeln
    candidates = []
    for ext in IMG_EXTS:
        candidates.extend(glob(os.path.join(IMAGE_DIR, f"*{ext}")))
    candidates = [p for p in candidates if is_image_file(p)]
    if not candidates:
        print(f"Keine Bilder in {IMAGE_DIR} gefunden.")
        return

    # Paare bilden
    pairs = []
    for img_path in candidates:
        base = basename_no_ext(img_path)
        mask_path = os.path.join(MASK_DIR, f"{base}_mask.png")
        if os.path.isfile(mask_path):
            pairs.append((img_path, mask_path))

    if not pairs:
        print(f"Keine gültigen Bild-Maske-Paare gefunden (erwartet *_mask.png in {MASK_DIR}).")
        return

    if LIMIT is not None:
        pairs = pairs[:LIMIT]

    if SHUFFLE:
        random.seed(SEED)
        random.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * TRAIN_RATIO))
    n_val   = int(round(n * VAL_RATIO))
    n_test  = n - n_train - n_val
    if n_test < 0:
        n_test = max(0, n - n_train - n_val)

    train = pairs[:n_train]
    val   = pairs[n_train:n_train+n_val]
    test  = pairs[n_train+n_val:]

    def write_list(path, items, write_pairs=False, without_ext=True):
        with open(path, "w", encoding="utf-8") as f:
            for img_path, mask_path in items:
                if write_pairs:
                    f.write(f"{img_path}\t{mask_path}\n")
                else:
                    name = basename_no_ext(img_path) if without_ext else os.path.basename(img_path)
                    f.write(name + "\n")

    # Nur Basename OHNE Endung
    write_list(os.path.join(LISTS_DIR, "train.txt"), train, write_pairs=False, without_ext=True)
    write_list(os.path.join(LISTS_DIR, "val.txt"),   val,   write_pairs=False, without_ext=True)
    write_list(os.path.join(LISTS_DIR, "test.txt"),  test,  write_pairs=False, without_ext=True)

    # Optional zusätzlich: Pfad-Paare mitschreiben
    # write_list(os.path.join(LISTS_DIR, "train_pairs.tsv"), train, write_pairs=True)
    # write_list(os.path.join(LISTS_DIR, "val_pairs.tsv"),   val,   write_pairs=True)
    # write_list(os.path.join(LISTS_DIR, "test_pairs.tsv"),  test,  write_pairs=True)

    print(f"Gefundene Paare: {n}")
    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")

if __name__ == "__main__":
    main()
