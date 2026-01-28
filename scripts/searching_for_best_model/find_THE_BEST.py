import ast
from pathlib import Path
import pandas as pd

# === EINSTELLUNGEN ===
base_dir = Path("../../model_out")   # Wurzelordner, wird rekursiv durchsucht
fake_patterns = ["*val_metric_fake_epoch*.csv"]
real_patterns = ["*val_metric_real_epoch*.csv"]

dice_col_name = "mean_i_soft_dice"            # Spalte im Fake-CSV
cm_col_hint = "mean_confusion_matrix_bin"     # Spalte im Real-CSV (kann leicht abweichen)
top_k = 20
out_txt = Path("top_models_summary.txt")
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

def find_col(df: pd.DataFrame, key_like: str):
    """
    Finde Spalte robust (case-insensitive, trim), erlaubt Teiltreffer.
    """
    low_map = {c.strip().lower(): c for c in df.columns}
    target_l = key_like.strip().lower()
    if target_l in low_map:
        return low_map[target_l]
    # Teiltreffer
    for k, orig in low_map.items():
        if target_l in k:
            return orig
    return None

def best_dice_from_fake_csv(csv_path: Path):
    """
    Liefert bestes (höchstes) Dice pro CSV-Datei.
    """
    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        return None

    col = find_col(df, dice_col_name)
    if not col:
        return None

    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return None

    idx = s.idxmax()
    val = float(s.loc[idx])
    # epoch ermitteln, wenn vorhanden
    ep = None
    if "epoch" in df.columns and idx in df.index:
        try:
            ep = int(df.loc[idx, "epoch"])
        except Exception:
            ep = None

    return {
        "value": val,
        "epoch": ep,
        "row_index": idx,
        "csv_path": csv_path.resolve(),
        "dir_path": csv_path.resolve().parent
    }

def parse_confusion_matrix_cell(cell):
    """
    Erwartet String/List der Form [[tp, fp],[fn, tn]] oder flach [tp, fp, fn, tn].
    Gibt (tp, fp, fn, tn) als floats zurück oder None bei Fehler.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    try:
        # String -> Python-Objekt
        if isinstance(cell, str):
            obj = ast.literal_eval(cell)
        else:
            obj = cell
        # 2x2 oder flach
        if isinstance(obj, (list, tuple)):
            if len(obj) == 2 and all(isinstance(x, (list, tuple)) for x in obj):
                # [[tp, fp],[fn, tn]]
                tp, fp = float(obj[0][0]), float(obj[0][1])
                fn, tn = float(obj[1][0]), float(obj[1][1])
                return tp, fp, fn, tn
            elif len(obj) == 4:
                # [tp, fp, fn, tn]
                tp, fp, fn, tn = map(float, obj)
                return tp, fp, fn, tn
    except Exception:
        return None
    return None

def best_fpr_from_real_csv(csv_path: Path):
    """
    Liefert bestes (niedrigstes) FPR pro CSV-Datei aus mean-confusion_matrix_bin.
    FPR = FP / (FP + TN).
    """
    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        return None

    cm_col = find_col(df, cm_col_hint)
    if not cm_col:
        # Fallback: jede Spalte, die 'confusion' und 'bin' enthält
        cm_col = None
        for c in df.columns:
            lc = c.strip().lower()
            if "confusion" in lc and "bin" in lc:
                cm_col = c
                break
        if not cm_col:
            return None

    fprs = []
    for idx, cell in df[cm_col].items():
        parsed = parse_confusion_matrix_cell(cell)
        if not parsed:
            continue
        tp, fp, fn, tn = parsed
        denom = fp + tn
        if denom <= 0:
            continue
        fpr = float(fp / denom)
        # epoch wenn vorhanden
        ep = None
        if "epoch" in df.columns and idx in df.index:
            try:
                ep = int(df.loc[idx, "epoch"])
            except Exception:
                ep = None
        fprs.append({
            "fpr": fpr,
            "epoch": ep,
            "row_index": idx,
            "csv_path": csv_path.resolve(),
            "dir_path": csv_path.resolve().parent
        })

    if not fprs:
        return None

    # bestes (kleinstes) FPR in dieser Datei
    best = min(fprs, key=lambda d: d["fpr"])
    return best

def main():
    # --- Fake/Dice sammeln ---
    dice_candidates = []
    for csv_path in find_csvs(base_dir, fake_patterns):
        res = best_dice_from_fake_csv(csv_path)
        if res:
            dice_candidates.append(res)

    # --- Real/FPR sammeln ---
    fpr_candidates = []
    for csv_path in find_csvs(base_dir, real_patterns):
        res = best_fpr_from_real_csv(csv_path)
        if res:
            fpr_candidates.append(res)

    if not dice_candidates and not fpr_candidates:
        print(f"Keine passenden CSVs unter {base_dir} gefunden.")
        return

    dice_candidates_sorted = sorted(dice_candidates, key=lambda x: x["value"], reverse=True)[:top_k]
    fpr_candidates_sorted  = sorted(fpr_candidates,  key=lambda x: x["fpr"])[:top_k]

    # --- TXT schreiben ---
    lines = []
    lines.append("== Top Dice auf Fake (höher ist besser) ==\n")
    if dice_candidates_sorted:
        for i, r in enumerate(dice_candidates_sorted, 1):
            ep = f"epoch={r['epoch']}" if r["epoch"] is not None else "epoch=?"
            lines.append(f"{i:>2}. dice={r['value']:.6f} | {ep}")
            lines.append(f"    dir: {r['dir_path']}")
            lines.append(f"    csv: {r['csv_path']}\n")
    else:
        lines.append("Keine Dice-Ergebnisse gefunden.\n")

    lines.append("\n== Top FPR auf Real (niedriger ist besser) ==\n")
    if fpr_candidates_sorted:
        for i, r in enumerate(fpr_candidates_sorted, 1):
            ep = f"epoch={r['epoch']}" if r["epoch"] is not None else "epoch=?"
            lines.append(f"{i:>2}. fpr={r['fpr']:.8f} | {ep}")
            lines.append(f"    dir: {r['dir_path']}")
            lines.append(f"    csv: {r['csv_path']}\n")
    else:
        lines.append("Keine FPR-Ergebnisse gefunden.\n")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    # --- Konsolen-Output kurz halten ---
    print(f"[OK] Geschrieben: {out_txt.resolve()}")
    if dice_candidates_sorted:
        print(f"Top Dice: {dice_candidates_sorted[0]['value']:.6f} @ {dice_candidates_sorted[0]['dir_path'].name}")
    if fpr_candidates_sorted:
        print(f"Top (min) FPR: {fpr_candidates_sorted[0]['fpr']:.8f} @ {fpr_candidates_sorted[0]['dir_path'].name}")

if __name__ == "__main__":
    main()
