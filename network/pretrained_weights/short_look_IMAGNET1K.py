import torch, types
from pathlib import Path

# Pfad zur .pth-Datei
ckpt_path = Path("./swin_b-68c6b09e.pth")
output_txt = Path("./IMAGENET1K_structure.txt")

# --- Laden ---
sd = torch.load(ckpt_path, map_location="cpu")

# --- Struktur prüfen ---
if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
    state_dict = sd
    header_info = "[INFO] Datei enthält direkt ein state_dict.\n"
else:
    header_info = "[WARN] Datei enthält verschachtelte Struktur:\n"
    header_info += f"Top-Level-Keys: {list(sd.keys())}\n\n"
    if "model" in sd:
        state_dict = sd["model"]
    elif "state_dict" in sd:
        state_dict = sd["state_dict"]
    else:
        raise ValueError("Unbekannte Struktur im .pth-File – bitte Key anpassen.")

# --- Informationen sammeln ---
lines = []
lines.append(header_info)
lines.append(f"[INFO] Anzahl Parameter: {len(state_dict)}\n")

for i, (k, v) in enumerate(state_dict.items()):
    shape = getattr(v, "shape", None)
    dtype = getattr(v, "dtype", None)
    lines.append(f"{i:3d}: {k:60s}  {shape}  {dtype}")

# --- In Datei schreiben ---
output_txt.parent.mkdir(parents=True, exist_ok=True)
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[OK] Struktur wurde in '{output_txt}' gespeichert ({len(state_dict)} Einträge).")

