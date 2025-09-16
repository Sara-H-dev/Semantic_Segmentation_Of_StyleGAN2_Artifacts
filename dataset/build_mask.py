#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# ==== ADJUST SETTINGS HERE ===========
# path to xml : 
XML_PATH = "C:/Users/sheyd/Documents/9_Bachelorarbiet/Daten/Gelabelte Bilder/CVAT_for_images_1.1/Extrackted/merged_annotation_oldnames.xml"   
# folder where the masks are saved:
OUT_MASK_DIR = "./labels"        
# folder of the images
IMAGE_DIR = "C:/Users/sheyd/Documents/9_Bachelorarbiet/Daten/Gelabelte Bilder/CVAT_for_images_1.1/Extrackted/images"   
# folder where the images are getting copyed to   
OUT_IMAGE_DIR = "./images"      
# =====================================

def parse_points(points_str):
    """CVAT-Format: 'x1,y1;x2,y2;...' -> Liste [(x1,y1), (x2,y2), ...]"""
    pts = []
    for pair in points_str.strip().split(';'):
        if not pair:
            continue
        xy = pair.split(',')
        if len(xy) != 2:
            continue
        try:
            pts.append((float(xy[0]), float(xy[1])))
        except ValueError:
            continue
    return pts

def make_mask_for_image(width, height, polygons):
    """Erstellt eine Binärmaske (0=schwarz, 255=weiß)."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon(poly, fill=255, outline=255)
    return mask

def main():
    ap = argparse.ArgumentParser(description="CVAT XML -> Artefakt-Masken und passende Bilder kopieren")
    ap.add_argument("--limit", type=int, default=None,
                    help="Maximale Anzahl Images (in XML-Reihenfolge). Ohne Angabe: alle.")
    args = ap.parse_args()

    if not os.path.isfile(XML_PATH):
        print(f"XML nicht gefunden: {XML_PATH}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

    # XML parsen
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    images = root.findall(".//image")
    if not images:
        print("Keine <image>-Einträge gefunden!", file=sys.stderr)
        sys.exit(1)

    to_process = images if args.limit is None else images[:args.limit]
    print(f"Gefundene Images: {len(images)}. Verarbeite: {len(to_process)}")

    for i, img in enumerate(to_process, 1):
        name = img.get("name")
        width = int(float(img.get("width")))
        height = int(float(img.get("height")))

        # Polygone sammeln
        polys = []
        for poly in img.findall("./polygon"):
            if poly.get("label") == "Artefakt":
                pts = parse_points(poly.get("points", ""))
                if pts:
                    polys.append(pts)

        # Maske erstellen und speichern
        mask = make_mask_for_image(width, height, polys)
        base = os.path.splitext(os.path.basename(name))[0]
        out_mask_path = os.path.join(OUT_MASK_DIR, f"{base}_mask.png")
        mask.save(out_mask_path, "PNG")

        # Originalbild kopieren, falls vorhanden
        src_img_path = os.path.join(IMAGE_DIR, name)
        if os.path.isfile(src_img_path):
            dst_img_path = os.path.join(OUT_IMAGE_DIR, name)
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warnung: Originalbild nicht gefunden -> {src_img_path}", file=sys.stderr)

        print(f"[{i}/{len(to_process)}] Maske: {out_mask_path}")

    print("Fertig.")

if __name__ == "__main__":
    main()