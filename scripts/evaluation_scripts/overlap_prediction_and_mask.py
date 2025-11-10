#!/usr/bin/env python3
import os, re
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

# --- Pfade anpassen falls nötig ---
PRED_DIR = Path("../../model_out/LAST/test_071125_1240/predictions")
LABEL_DIR = Path("../../dataset/fake_labels")
OUT_DIR   = Path("final_overlays")  # Ergebnis-Ordner
THRESH = 0.5                        # für TP/FP/FN

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_png_gray01(path: Path) -> np.ndarray:
    """Lädt PNG als float32 in [0,1], grayscale."""
    img = Image.open(path).convert("L")  # 8-bit Grau
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def save_rgb01(arr: np.ndarray, path: Path):
    """Speichert [H,W,3] float32 in [0,1] als PNG."""
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).round().astype(np.uint8))
    img.save(path)

def resize_to(arr: np.ndarray, shape_hw: Tuple[int,int], is_mask: bool) -> np.ndarray:
    """Resize numpy-Bild auf shape_hw. is_mask steuert Interpolation."""
    H, W = shape_hw
    pil = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8))
    if is_mask:
        pil = pil.resize((W, H), resample=Image.NEAREST)
    else:
        pil = pil.resize((W, H), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return out

def extract_id6_from_pred_name(name: str) -> Optional[str]:
    """
    Versucht eine 6-stellige Nummer zu finden (z.B. 098000).
    Wenn nicht gefunden, nimmt die ersten 6 Ziffern ab Start.
    """
    m = re.search(r"(\d{6})", name)
    if m:
        return m.group(1)
    # fallback: nimm fortlaufende Ziffern ab Anfang und cut auf 6
    m2 = re.match(r"(\d+)", name)
    if m2 and len(m2.group(1)) >= 6:
        return m2.group(1)[:6]
    return None

def confusion_from_probs(pred01: np.ndarray, label01: np.ndarray, thr: float):
    bin_pred = pred01 >= thr
    lbl = label01 >= 0.5
    tp = np.logical_and(bin_pred, lbl)
    fp = np.logical_and(bin_pred, np.logical_not(lbl))
    fn = np.logical_and(np.logical_not(bin_pred), lbl)
    return tp, fp, fn

def dice(bin_pred: np.ndarray, label: np.ndarray, eps: float=1e-8) -> float:
    inter = np.logical_and(bin_pred, label).sum()
    denom = bin_pred.sum() + label.sum()
    return (2.0*inter + eps) / (denom + eps)

def overlay_pred_label(pred01: np.ndarray, label01: np.ndarray) -> np.ndarray:
    """
    RGB: R=pred heat, G=label, B=0  -> Gelb = Überlappung
    """
    H, W = label01.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[...,0] = pred01
    rgb[...,1] = label01
    # Optional: leichte Gamma-Anhebung für Sichtbarkeit
    # rgb = np.power(rgb, 0.9)
    return rgb

def overlay_tp_fp_fn(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """
    RGB: TP=grün, FP=rot, FN=blau. Mehrfachhits gemischt additiv geklammert.
    """
    H, W = tp.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[...,1] += tp.astype(np.float32)  # G
    rgb[...,0] += fp.astype(np.float32)  # R
    rgb[...,2] += fn.astype(np.float32)  # B
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

def process_one(pred_path: Path):
    # nur Dateien, die mit "09" beginnen und "_grey_heats.png" enden
    name = pred_path.name
    if not (name.startswith("09") and name.endswith("_grey_heats.png")):
        return

    id6 = extract_id6_from_pred_name(name)
    if id6 is None:
        print(f"[WARN] Konnte ID nicht aus {name} extrahieren – überspringe.")
        return

    label_path = LABEL_DIR / f"{id6}_mask.png"
    if not label_path.exists():
        print(f"[WARN] Label fehlt: {label_path} – überspringe {name}.")
        return

    # Laden
    pred01  = load_png_gray01(pred_path)       # heat/prob in [0,1]
    label01 = load_png_gray01(label_path)      # annimmt {0,1} (sonst wird gethresholded)

    # Größen abgleichen
    if pred01.shape != label01.shape:
        pred01 = resize_to(pred01, label01.shape, is_mask=False)

    # Visualisierung 1: pred vs label
    viz1 = overlay_pred_label(pred01, label01)
    #save_rgb01(viz1, OUT_DIR / f"{id6}_A_pred_label.png")

    # Visualisierung 2: TP/FP/FN (mit Threshold)
    tp, fp, fn = confusion_from_probs(pred01, label01, THRESH)
    viz2 = overlay_tp_fp_fn(tp, fp, fn)
    save_rgb01(viz2, OUT_DIR / f"{id6}_B_tp_fp_fn_thr{int(THRESH*100)}.png")

    # Kennzahlen
    bin_pred = pred01 >= THRESH
    lbl = label01 >= 0.5
    d = dice(bin_pred, lbl)
    H, W = lbl.shape
    area = float(H*W)
    fp_pct_img = 100.0 * fp.sum() / area
    fn_pct_lbl = 100.0 * (fn.sum() / max(1, lbl.sum()))
    """with open(OUT_DIR / f"{id6}_metrics.txt", "w") as f:
        f.write(f"id={id6}\n")
        f.write(f"pred_file={pred_path.name}\n")
        f.write(f"label_file={label_path.name}\n")
        f.write(f"threshold={THRESH}\n")
        f.write(f"dice={d:.4f}\n")
        f.write(f"FP_percent_of_image={fp_pct_img:.4f}%\n")
        f.write(f"FN_percent_of_label={fn_pct_lbl:.4f}%\n")
        f.write(f"TP_pixels={tp.sum()}, FP_pixels={fp.sum()}, FN_pixels={fn.sum()}\n")"""

def main():
    files = sorted(PRED_DIR.glob("09*_grey_heats.png"))
    if not files:
        print(f"[INFO] Keine passenden Dateien in {PRED_DIR} gefunden.")
    for p in files:
        try:
            process_one(p)
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

if __name__ == "__main__":
    main()
