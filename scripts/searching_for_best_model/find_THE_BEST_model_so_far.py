import pandas as pd
from pathlib import Path

# === EINSTELLUNGEN ===
base_dir = Path("../model_out")  # Wurzelordner, wird rekursiv durchsucht
patterns = [
    "*val_metric_fake_epoch*.csv"
]
target_col = "mean_i_soft_dice"
top_k = 5
# ======================

def find_csvs(base: Path, patterns):
    files = set()
    for pat in patterns:
        files.update(base.rglob(pat))  # rekursiv!
    return sorted(files)

def safe_read_csv(p: Path):
    try:
        return pd.read_csv(p, on_bad_lines="skip")
    except Exception as e:
        print(f"[WARN] Konnte {p} nicht lesen: {e}")
        return None

def get_best_from_df(df: pd.DataFrame, col_name: str):
    # Spalten robuster behandeln: Whitespace entfernen, case-insensitive
    clean_map = {c.strip().lower(): c for c in df.columns}
    key = col_name.strip().lower()
    if key not in clean_map:
        return None

    col = clean_map[key]
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return None

    idx = s.idxmax()
    val = s.loc[idx]
    if pd.isna(val):
        return None

    # epoch (falls vorhanden)
    epoch = None
    if "epoch" in df.columns and idx in df.index:
        try:
            epoch = df.loc[idx, "epoch"]
        except Exception:
            epoch = None

    return {"row_index": idx, "value": float(val), "epoch": epoch}

def main():
    csv_files = find_csvs(base_dir, patterns)
    if not csv_files:
        print(f"Keine CSVs unter {base_dir} zu Mustern {patterns} gefunden.")
        return

    best = None
    candidates = []

    for csv_path in csv_files:
        df = safe_read_csv(csv_path)
        if df is None:
            continue

        res = get_best_from_df(df, target_col)
        if not res:
            continue

        record = {
            "value": res["value"],
            "csv_path": csv_path.resolve(),
            "dir_path": csv_path.resolve().parent,
            "row_index": res["row_index"],
            "epoch": res["epoch"],
        }
        candidates.append(record)

        if (best is None) or (res["value"] > best["value"]):
            best = record

    if not candidates:
        print(f"Keine gültigen '{target_col}'-Werte gefunden.")
        return

    # Ergebnis ausgeben
    print("== Bestes mean_i_soft_dice über alle Unterordner ==")
    print(f"Wert:        {best['value']:.6f}")
    print(f"CSV-Datei:   {best['csv_path']}")
    print(f"Ordner:      {best['dir_path']}")
    print(f"Zeilenindex: {best['row_index']}")
    if best["epoch"] is not None:
        print(f"Epoch:       {best['epoch']}")

    # Top-K Übersicht
    print(f"\n== Top {top_k} Dateien (nach Wert) ==")
    for i, r in enumerate(sorted(candidates, key=lambda x: x["value"], reverse=True)[:top_k], start=1):
        ep = f", epoch={r['epoch']}" if r["epoch"] is not None else ""
        print(f"{i:>2}. {r['value']:.6f}  | {r['dir_path'].name}{ep}")
        print(f"    {r['csv_path']}")

if __name__ == "__main__":
    main()
