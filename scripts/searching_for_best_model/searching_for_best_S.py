import ast
from pathlib import Path
import pandas as pd

# =================== EINSTELLUNGEN ===================
base_dir = Path("../../model_out")     # Wurzelordner, rekursiv durchsuchen
fake_patterns = ["*val_metric_fake_epoch*.csv"]
real_patterns = ["*val_metric_real_epoch*.csv"]

dice_col_name = "mean_i_soft_dice"            # Spalte im Fake-CSV
cm_col_hint  = "mean_confusion_matrix_bin"    # Spalte im Real-CSV ([[tp, fp],[fn, tn]])
W = 0.7                                      # Gewicht w in S = w*Dice + (1-w)*(1-FPR)
TOP_K = 20                                    # wie viele Top-Ergebnisse in die TXT
OUT_TXT = Path("top_models_by_S.txt")
# =====================================================

def find_csvs(base: Path, patterns):
    files = set()
    for pat in patterns:
        files.update(base.rglob(pat))
    return sorted(files)

def safe_read_csv(p: Path):
    try:
        return pd.read_csv(p, on_bad_lines="skip")
    except Exception as e:
        print(f"[WARN] Konnte {p} nicht lesen: {e}")
        return None

def find_col(df: pd.DataFrame, key_like: str):
    """Case-insensitive, trim, erlaubt Teiltreffer."""
    low_map = {c.strip().lower(): c for c in df.columns}
    target_l = key_like.strip().lower()
    if target_l in low_map:
        return low_map[target_l]
    for k, orig in low_map.items():
        if target_l in k:
            return orig
    return None

def parse_confusion_matrix_cell(cell):
    """Erwartet [[tp, fp],[fn, tn]] oder [tp, fp, fn, tn]; gibt (tp,fp,fn,tn) als floats zurück."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    try:
        obj = ast.literal_eval(cell) if isinstance(cell, str) else cell
        if isinstance(obj, (list, tuple)):
            if len(obj) == 2 and all(isinstance(x, (list, tuple)) for x in obj):
                tp, fp = float(obj[0][0]), float(obj[0][1])
                fn, tn = float(obj[1][0]), float(obj[1][1])
                return tp, fp, fn, tn
            elif len(obj) == 4:
                tp, fp, fn, tn = map(float, obj)
                return tp, fp, fn, tn
    except Exception:
        return None
    return None

def load_fake_metrics(csv_path: Path):
    """
    Lädt Fake-CSV und gibt DataFrame mit Spalten ['epoch','dice_fake'] zurück.
    Falls 'epoch' fehlt, wird der Zeilenindex als 'epoch' verwendet.
    """
    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        return None
    col_dice = find_col(df, dice_col_name)
    if not col_dice:
        return None
    fake = pd.DataFrame()
    fake["dice_fake"] = pd.to_numeric(df[col_dice], errors="coerce")
    fake["epoch"] = df["epoch"] if "epoch" in df.columns else df.index
    fake = fake.dropna(subset=["dice_fake"]).reset_index(drop=True)
    return fake

def load_real_fpr(csv_path: Path):
    """
    Lädt Real-CSV und baut DataFrame mit Spalten ['epoch','fpr_real'].
    FPR = FP / (FP + TN) aus mean-confusion_matrix_bin.
    """
    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        return None

    cm_col = find_col(df, cm_col_hint)
    if not cm_col:
        # Fallback: erste Spalte, die 'confusion' und 'bin' enthält
        for c in df.columns:
            lc = c.strip().lower()
            if "confusion" in lc and "bin" in lc:
                cm_col = c
                break
        if not cm_col:
            return None

    rows = []
    for idx, cell in df[cm_col].items():
        parsed = parse_confusion_matrix_cell(cell)
        if not parsed:
            continue
        tp, fp, fn, tn = parsed
        denom = fp + tn
        if denom <= 0:
            continue
        fpr = float(fp / denom)
        ep = df.loc[idx, "epoch"] if "epoch" in df.columns else idx
        rows.append({"epoch": int(ep), "fpr_real": fpr})
    if not rows:
        return None
    real = pd.DataFrame(rows).reset_index(drop=True)
    # bei mehrfachen Einträgen pro epoch aggregieren (Minimum der FPR bevorzugen)
    real = real.groupby("epoch", as_index=False)["fpr_real"].min()
    return real

def collect_runs(base_dir: Path):
    """
    Sucht in jedem Unterordner Fake- und Real-CSV.
    Baut pro Ordner den besten S-Score + Epoche.
    """
    # Index nach Ordnerpfad
    fake_map = {}
    real_map = {}

    for p in find_csvs(base_dir, fake_patterns):
        fake_map.setdefault(p.parent.resolve(), []).append(p)
    for p in find_csvs(base_dir, real_patterns):
        real_map.setdefault(p.parent.resolve(), []).append(p)

    common_dirs = sorted(set(fake_map.keys()) & set(real_map.keys()))
    results = []

    for d in common_dirs:
        # Lade und merge alle Fake/Real CSVs in dem Ordner (falls mehrere, werden sie zusammengeführt)
        fake_dfs = [load_fake_metrics(p) for p in fake_map[d]]
        real_dfs = [load_real_fpr(p)    for p in real_map[d]]
        fake_dfs = [x for x in fake_dfs if x is not None and not x.empty]
        real_dfs = [x for x in real_dfs if x is not None and not x.empty]
        if not fake_dfs or not real_dfs:
            continue

        fake_all = pd.concat(fake_dfs, ignore_index=True)
        real_all = pd.concat(real_dfs, ignore_index=True)

        # Auf Epoche joinen (inner join: nur Epochen, die beide Metriken haben)
        merged = pd.merge(fake_all, real_all, on="epoch", how="inner")
        if merged.empty:
            continue

        # Score S berechnen
        # merged["S"] = W*merged["dice_fake"] + (1.0 - W)*(1.0 - merged["fpr_real"])
        merged["S"] = hä merged["dice_fake"] - 5 * merged["fpr_real"]

        # Beste Epoche pro Ordner
        j = int(merged["S"].idxmax())
        best_row = merged.loc[j]
        results.append({
            "dir": d,
            "epoch": int(best_row["epoch"]),
            "dice_fake": float(best_row["dice_fake"]),
            "fpr_real": float(best_row["fpr_real"]),
            "S": float(best_row["S"])
        })

    return results

def main():
    res = collect_runs(base_dir)
    if not res:
        print(f"Keine kombinierten (Fake/Real) Ergebnisse unter {base_dir} gefunden.")
        return

    df = pd.DataFrame(res).sort_values("S", ascending=False).reset_index(drop=True)

    # TXT ausgeben
    lines = []
    lines.append(f"Score: S = {W:.3f}*Dice_fake + {1-W:.3f}*(1 - FPR_real)\n")
    lines.append("== Top Modelle nach S (höher ist besser) ==\n")
    for i, row in df.head(TOP_K).iterrows():
        lines.append(f"{i+1:>2}. S={row['S']:.6f} | epoch={row['epoch']} | "
                     f"dice={row['dice_fake']:.6f} | fpr={row['fpr_real']:.8f}")
        lines.append(f"    dir: {row['dir']}\n")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Geschrieben: {OUT_TXT.resolve()}")

    # Kurz zusammenfassen
    best = df.iloc[0]
    print(f"Bestes Modell:\n  S={best['S']:.6f} | epoch={best['epoch']} | "
          f"dice={best['dice_fake']:.6f} | fpr={best['fpr_real']:.8f}\n  dir={best['dir']}")

if __name__ == "__main__":
    main()
