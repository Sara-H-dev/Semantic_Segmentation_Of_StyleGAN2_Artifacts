import ast
from pathlib import Path
import pandas as pd

# === EINSTELLUNGEN ===
base_dir = Path("../../model_out")
fake_patterns = ["*val_metric_fake_epoch*.csv"]
real_patterns = ["*val_metric_real_epoch*.csv"]
cm_col_hint = "mean_confusion_matrix_bin"

top_k = 20
w_real = 0.2   # Gewichtung Real
w_fake = 1 - w_real   # Gewichtung Fake
out_txt = Path("top_weighted_accuracy_summary.txt")
# ======================


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
    low_map = {c.strip().lower(): c for c in df.columns}
    target_l = key_like.strip().lower()
    if target_l in low_map:
        return low_map[target_l]
    for k, orig in low_map.items():
        if target_l in k:
            return orig
    return None


def parse_confusion_matrix_cell(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    try:
        if isinstance(cell, str):
            obj = ast.literal_eval(cell)
        else:
            obj = cell
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


def get_best_acc(csv_path: Path, weighted_tp=False):
    """Gibt beste Accuracy (normal oder weighted) zurück"""
    df = safe_read_csv(csv_path)
    if df is None or df.empty:
        return None

    cm_col = find_col(df, cm_col_hint)
    if not cm_col:
        return None

    accs = []
    for idx, cell in df[cm_col].items():
        parsed = parse_confusion_matrix_cell(cell)
        if not parsed:
            continue
        tp, fp, fn, tn = parsed
        if weighted_tp:
            denom = (2 * tp + tn + fp + fn)
            acc = (2 * tp + tn) / denom if denom > 0 else None
        else:
            denom = (tp + tn + fp + fn)
            acc = (tp + tn) / denom if denom > 0 else None
        if acc is None:
            continue

        ep = None
        if "epoch" in df.columns and idx in df.index:
            try:
                ep = int(df.loc[idx, "epoch"])
            except Exception:
                ep = None

        accs.append({
            "acc": acc,
            "epoch": ep,
            "csv_path": csv_path.resolve(),
            "dir_path": csv_path.resolve().parent
        })

    if not accs:
        return None
    return max(accs, key=lambda d: d["acc"])


def main():
    # --- Fake Weighted Accuracy ---
    fake_accs = []
    for csv_path in find_csvs(base_dir, fake_patterns):
        res = get_best_acc(csv_path, weighted_tp=True)
        if res:
            fake_accs.append(res)

    # --- Real Normal Accuracy ---
    real_accs = []
    for csv_path in find_csvs(base_dir, real_patterns):
        res = get_best_acc(csv_path, weighted_tp=False)
        if res:
            real_accs.append(res)

    if not fake_accs and not real_accs:
        print(f"Keine passenden CSVs unter {base_dir} gefunden.")
        return

    # Map Dir -> Acc für Kombination
    fake_by_dir = {r["dir_path"].name: r for r in fake_accs}
    real_by_dir = {r["dir_path"].name: r for r in real_accs}

    combined = []
    for dir_name, real_entry in real_by_dir.items():
        fake_entry = fake_by_dir.get(dir_name)
        if not fake_entry:
            continue
        score = w_real * real_entry["acc"] + w_fake * fake_entry["acc"]
        combined.append({
            "score": score,
            "acc_real": real_entry["acc"],
            "acc_fake": fake_entry["acc"],
            "epoch_real": real_entry["epoch"],
            "epoch_fake": fake_entry["epoch"],
            "dir_path": real_entry["dir_path"],
        })

    combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]
    fake_sorted = sorted(fake_accs, key=lambda x: x["acc"], reverse=True)[:top_k]
    real_sorted = sorted(real_accs, key=lambda x: x["acc"], reverse=True)[:top_k]

    # --- TXT schreiben ---
    lines = []
    lines.append(f"== Kombinierter Score (w_real={w_real}, w_fake={w_fake}) ==\n")
    if combined_sorted:
        for i, r in enumerate(combined_sorted, 1):
            lines.append(f"{i:>2}. score={r['score']:.6f} | real={r['acc_real']:.6f}, fake={r['acc_fake']:.6f}")
            lines.append(f"    epochs: real={r['epoch_real']}, fake={r['epoch_fake']}")
            lines.append(f"    dir: {r['dir_path']}\n")
    else:
        lines.append("Keine gemeinsamen Real/Fake-Verzeichnisse gefunden.\n")

    lines.append("\n== Top Weighted Accuracy auf Fake ==\n")
    for i, r in enumerate(fake_sorted, 1):
        ep = f"epoch={r['epoch']}" if r['epoch'] is not None else "epoch=?"
        lines.append(f"{i:>2}. acc={r['acc']:.6f} | {ep}")
        lines.append(f"    dir: {r['dir_path']}\n")

    lines.append("\n== Top Accuracy auf Real ==\n")
    for i, r in enumerate(real_sorted, 1):
        ep = f"epoch={r['epoch']}" if r['epoch'] is not None else "epoch=?"
        lines.append(f"{i:>2}. acc={r['acc']:.6f} | {ep}")
        lines.append(f"    dir: {r['dir_path']}\n")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Ergebnisse gespeichert in: {out_txt.resolve()}")
    if combined_sorted:
        print(f"Top Combined Score: {combined_sorted[0]['score']:.6f} @ {combined_sorted[0]['dir_path'].name}")


if __name__ == "__main__":
    main()
