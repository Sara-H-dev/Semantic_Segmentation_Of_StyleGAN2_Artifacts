
import os
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cv2

def overlay(image_batch, heat, binmsk, pred_dir, image_pre, case_names):
    img  = image_batch[image_pre].detach().float().cpu()    # (3,H,W)
    img_name = case_names[image_pre]
    hm   = heat[image_pre, 0].detach().float().cpu()        # (H,W)
    alpha = 0.4
    overlay = img.clone()
    overlay[0] = torch.clamp(img[0]*(1-alpha) + alpha*hm, 0, 1)  # Rotkanal
    overlay[1] = img[1]*(1-alpha)
    overlay[2] = img[2]*(1-alpha)
    save_image(overlay, os.path.join(
        pred_dir, f"{img_name}_bin_overlay.png"))
    
def save_color_heatmap(img_3chw, heat_hw, out_png, alpha=0.4):
        """
        img_3chw:  (3,H,W) Tensor in [0,1] (denorm if necessary)
        heat_hw:   (H,W)   Tensor in [0,1] (sigmoid output)
        out_png:   Path
        alpha:     Overlay strength
        denorm:    optional callable: x -> x_denorm (for normalised inputs)
        """
        # 1) nach CPU/Numpy
        denorm=None
        img = img_3chw.detach().float().cpu()
        if denorm is not None:
            img = denorm(img)
        img = img.clamp(0,1).permute(1,2,0).numpy()  # (H,W,3)

        heat = heat_hw.detach().float().cpu().clamp(0,1).numpy()  # (H,W)

        # 2) Grün->Gelb->Rot Colormap (0=grün, 1=rot)
        cmap = LinearSegmentedColormap.from_list("g2r", [
            (0.0, "green"),
            (0.5, "yellow"),
            (1.0, "red"),
        ])

        # 3) Overlay (ohne Colorbar)
        hm_rgb = cmap(heat)[..., :3]                # (H,W,3), 0..1
        overlay = (1 - alpha) * img + alpha * hm_rgb
        overlay = np.clip(overlay, 0, 1)

        # 4) Plot mit Colorbar (Legende)
        fig, ax = plt.subplots(figsize=(6,6), dpi=200)
        ax.imshow(overlay)
        ax.set_axis_off()

        # „Mappable“ für Colorbar erzeugen (gleiche Skala für alle Bilder!)
        im = ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0)  # alpha=0 -> unsichtbar, nur für Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Artifact probability', rotation=270, labelpad=14)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['low', 'mid', 'high'])

        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)

def save_contour_heatmap(img_3chw, heat_hw, out_png):
    """
    img_3chw: (3,H,W) Tensor in [0,1]
    heat_hw:  (H,W) Tensor in [0,1]
    out_png:  Pfad zur Ausgabe
    """
    # --- 1) nach CPU/Numpy ---
    img = img_3chw.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()
    heat = heat_hw.detach().float().cpu().clamp(0,1).numpy()

    # --- 2) Colormap für Konturen (gleiche wie vorher, Grün-Gelb-Rot) ---
    cmap = LinearSegmentedColormap.from_list("g2r", [
        (0.0, "green"),
        (0.5, "yellow"),
        (1.0, "red"),
    ])

    # --- 3) Plot ---
    fig, ax = plt.subplots(figsize=(6,6), dpi=200)
    ax.imshow(img)
    # Konturen bei bestimmten Schwellen zeichnen
    contour_levels = [0.3, 0.6, 0.9]
    cs = ax.contour(heat, levels=contour_levels, cmap=cmap, linewidths=1.5)

    # Optional: Labels an Konturen
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f')

    ax.set_axis_off()

    # --- 4) Colorbar ---
    # unsichtbares heatmap-Image nur für die Colorbar
    im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, alpha=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Artifact probability', rotation=270, labelpad=14)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['low', 'mid', 'high'])

    # --- 5) Speichern ---
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

def overlay_mask_on_image(img_path, mask_path, out_path=None, color=(255, 0, 255), alpha=0.25, border_thickness=2):
    """
    img_path: Pfad zum Originalbild (RGB)
    mask_path: Pfad zur binären Maske (0/255 oder 0/1)
    out_path: optionaler Speicherpfad
    color: Farbe der Maske (RGB)
    alpha: Transparenz der Überlagerung (je kleiner, desto transparenter)
    border_thickness: Dicke der farbigen Umrandung
    """
    # Bild und Maske laden
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 127).astype(np.uint8)

    # Konturen berechnen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Farbmaske erzeugen
    overlay = img.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    # Fläche (transparent)
    mask_3ch = np.stack([mask]*3, axis=-1)
    overlay = np.where(mask_3ch > 0,
                       overlay * (1 - alpha) + color_arr * alpha,
                       overlay)

    # Umrandung zeichnen
    overlay = overlay.astype(np.uint8)
    cv2.drawContours(overlay, contours, -1, color, thickness=border_thickness)

    # Speichern oder anzeigen
    if out_path:
        Image.fromarray(overlay).save(out_path)
    else:
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()

def create_bin_heat_mask_from_list(ten_output_saver, pred_dir, dataset_root):

    os.makedirs(pred_dir, exist_ok=True)

    for case_name, pred_tensor in ten_output_saver:
        case_name = str(case_name)
        pred_tensor  = pred_tensor.detach().cpu()

        # load original image:
        if case_name.startswith("09"):
            img_path = os.path.join(dataset_root, "fake_images", f"{case_name}.png")
        else:
            img_path = os.path.join(dataset_root, "real_images", f"{case_name}.png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0

        if pred_tensor.ndim == 4: pred_tensor = pred_tensor[0] 
            
        heat   = pred_tensor.clamp(0, 1)         # in [0,1]
        binmsk = (heat > 0.4).float()

        save_image(heat, os.path.join(pred_dir, f"{case_name}_grey_heats.png"))
        save_image(binmsk, os.path.join(pred_dir, f"{case_name}_bin_mask.png"))
        image.save(os.path.join(pred_dir, f"{case_name}.png"))

        save_color_heatmap(
            img_3chw=image_tensor,
            heat_hw=heat[0] if heat.ndim == 3 else heat,
            out_png=os.path.join(pred_dir, f"{case_name}_heatmap.png"),
            alpha= 0.45 )
        
        overlay_mask_on_image(
            img_path= img_path,
            mask_path= os.path.join(pred_dir, f"{case_name}_bin_mask.png"),
            out_path=os.path.join(pred_dir, f"{case_name}_overlay_color.png"),
            color=(255, 0, 255),
            alpha= 0.25,
            border_thickness=2)