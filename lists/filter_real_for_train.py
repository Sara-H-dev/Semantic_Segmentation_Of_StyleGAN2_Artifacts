# create_real_train_all.py
# Liest alle Bilddateien aus ./real_images
# und schreibt deren Nummern (ohne .png)
# in real_train_all.txt, falls sie NICHT in val.txt oder test.txt vorkommen.

import os

# === Pfade ===
real_dir = "../dataset/real_images"
val_file = "val.txt"
test_file = "test.txt"
output_file = "real_train_all.txt"

# === Hilfsfunktion zum Einlesen der IDs aus val/test ===
def load_ids_from_txt(path):
    ids = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # falls Zeilen wie "123.png" oder "real_images/123.png" drinstehen
                basename = os.path.basename(line)
                name, _ = os.path.splitext(basename)
                ids.add(name)
    return ids

# === val/test IDs laden ===
val_ids = load_ids_from_txt(val_file)
test_ids = load_ids_from_txt(test_file)
exclude_ids = val_ids.union(test_ids)

# === real_images durchsuchen ===
all_real_ids = []
for filename in os.listdir(real_dir):
    if filename.lower().endswith(".png"):
        name, _ = os.path.splitext(filename)
        if name not in exclude_ids:
            all_real_ids.append(name)

# === Ausgabe schreiben ===
with open(output_file, "w") as out:
    for img_id in sorted(all_real_ids):
        out.write(img_id + "\n")

print(f"✅ {len(all_real_ids)} IDs gespeichert in {output_file}")
