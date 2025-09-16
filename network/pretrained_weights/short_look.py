import torch, types

ckpt_path = ".\SegFace_swin_celaba_512.pt"
# Checkpoint laden
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Top-Level-Keys:", list(ckpt.keys()))

# Backbone-State-Dict herausziehen
sd = ckpt["state_dict_backbone"]
print(f"\n[INFO] Backbone-State-Dict mit {len(sd)} Keys\n")

# Einige Keys und Shapes ausgeben
for i, (k, v) in enumerate(sd.items()):
    shape = getattr(v, "shape", None)
    print(f"{i:3d}: {k:60s} {shape}")