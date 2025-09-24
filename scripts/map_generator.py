
import os
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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