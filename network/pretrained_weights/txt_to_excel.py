#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vergleicht MSUNet- und SegFace-Parameterzeilen (Name + torch.Size) zeilenweise
und schreibt eine Excel-Datei mit dem Vergleich.
"""

import re
import pandas as pd
from pathlib import Path

# ====== PFAD-EINSTELLUNGEN (anpassen) ======
MSUNET_TXT   = Path(r"./structure_of_MSUNet.txt")
SEGFACE_TXT  = Path(r"./structure_of_SegFace.txt")
OUT_XLSX     = Path(r"./msunet_segface_compare.xlsx")
# ===========================================

PATTERN = re.compile(r'^\s*(\d+)\s*:\s*([^\s,]+).*?torch\.Size\(\[([^\]]*)\]\)', re.IGNORECASE)

def parse_param_file(path: Path):
    """Liest eine Struktur-Textdatei und gibt Liste von Dicts mit (idx, name, size_list, size_str) zurück."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    entries = []
    for line in lines:
        m = PATTERN.search(line)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2)
        size_str = m.group(3).strip()
        # Größe in Liste umwandeln (ints, wenn möglich)
        dims_raw = [s.strip() for s in size_str.split(",") if s.strip()]
        dims = []
        for d in dims_raw:
            try:
                dims.append(int(d))
            except ValueError:
                dims.append(d)  # Falls etwas Unerwartetes steht
        entries.append({
            "idx": idx,
            "name": name,
            "size_list": tuple(dims),
            "size_str": f"[{', '.join(map(str, dims))}]"
        })
    # Nach idx sortieren (sicherheitshalber)
    entries.sort(key=lambda x: x["idx"])
    return entries

def main():
    ms_entries  = parse_param_file(MSUNET_TXT)
    sf_entries  = parse_param_file(SEGFACE_TXT)
    max_len     = max(len(ms_entries), len(sf_entries))

    rows = []
    for i in range(max_len):
        ms = ms_entries[i] if i < len(ms_entries) else None
        sf = sf_entries[i] if i < len(sf_entries) else None

        ms_name = ms["name"] if ms else ""
        ms_size = ms["size_str"] if ms else ""
        ms_dims = ms["size_list"] if ms else None

        sf_name = sf["name"] if sf else ""
        sf_size = sf["size_str"] if sf else ""
        sf_dims = sf["size_list"] if sf else None

        size_match = (ms is not None and sf is not None and ms_dims == sf_dims)
        error = ""
        if ms is None or sf is None:
            error = "Missing row"
        elif not size_match:
            error = "Size mismatch"

        rows.append({
            "Index": i,
            "MSUNet Name": ms_name,
            "MSUNet Size": ms_size,
            "SegFace Name": sf_name,
            "SegFace Size": sf_size,
            "Size Match": bool(size_match) if (ms and sf) else None,
            "Error": error
        })

    df = pd.DataFrame(rows)
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_XLSX, index=False)
    print(f"✅ Geschrieben: {OUT_XLSX.resolve()}  ({len(df)} Zeilen)")

if __name__ == "__main__":
    main()