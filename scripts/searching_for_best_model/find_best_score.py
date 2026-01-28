#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# Basisordner (ggf. anpassen)
BASE_DIR = Path("../../model_out")
# Welche Überordner berücksichtigen
TARGET_TOPS = {"DROP", "DROP2", "DROP3"}
# Name der Ausgabedatei
OUT_TXT = BASE_DIR / "scores_summary.txt"

def find_target_dirs(base: Path):
    """Finde alle Verzeichnisse, die genau DROP, DROP2 oder DROP3 heißen (rekursiv)."""
    for p in base.rglob("*"):
        if p.is_dir() and p.name in TARGET_TOPS:
            yield p

def pick_fpr_col(df_real: pd.DataFrame):
    """Ermittle die FPR/FRP-Spalte robust."""
    for col in ["FRP", "FPR", "frp", "fpr"]:
        if col in df_real.columns:
            return col
    raise KeyError("Keine FRP/FPR-Spalte gefunden.")

def ensure_epoch_index(df: pd.DataFrame):
    """
    Stellt sicher, dass es eine 'epoch'-Spalte gibt.
    Falls nicht vorhanden, wird der Index als Epoche übernommen.
    """
    if "epoch" not in df.columns:
        df = df.copy()
        df["epoch"] = df.index
    return df

def summarize_run(run_dir: Path):
    """
    Liest die beiden CSVs eines Runs ein, merged sie auf 'epoch',
    berechnet Score je Epoche und gibt beste Zeile zurück.
    Rückgabe: dict oder None (falls unvollständig).
    """
    fake_csv = run_dir / "val_metric_fake_epoch.csv"
    real_csv = run_dir / "val_metric_real_epoch.csv"
    if not fake_csv.exists() or not real_csv.exists():
        return None  # unvollständig

    try:
        df_fake = pd.read_csv(fake_csv)
        df_real = pd.read_csv(real_csv)

        df_fake = ensure_epoch_index(df_fake)
        df_real = ensure_epoch_index(df_real)

        # benötigte Spalten prüfen
        if "mean_i_soft_dice" not in df_fake.columns:
            return None

        fpr_col = pick_fpr_col(df_real)

        # Mergen per Epoche (inner join, damit nur gemeinsame Epochen verglichen werden)
        df = pd.merge(
            df_fake[["epoch", "mean_i_soft_dice"]],
            df_real[["epoch", fpr_col]],
            on="epoch",
            how="inner",
        )

        if df.empty:
            return None

        # Score berechnen
        df["Score"] = df["mean_i_soft_dice"] - 10 * df[fpr_col]

        # beste Epoche (max Score)
        best = df.loc[df["Score"].idxmax()]

        return {
            "epoch": int(best["epoch"]),
            "score": float(best["Score"]),
            "fpr": float(best[fpr_col]),
            "mean_dice": float(best["mean_i_soft_dice"]),
        }
    except Exception:
        return None  # robust: Run überspringen bei Lesefehlern

def main():
    lines = []
    found_any = False

    for top in find_target_dirs(BASE_DIR):
        # alle direkten Unterordner des TOP-Ordners sind typischerweise die einzelnen Runs
        for run in sorted([p for p in top.iterdir() if p.is_dir()]):
            res = summarize_run(run)
            if res is None:
                continue

            found_any = True
            # „Überordner“ = Pfad relativ zu model_out, z. B. "DROP/run_name"
            rel = run.relative_to(BASE_DIR).as_posix()
            lines.append(rel)
            lines.append(
                f"(epoch, Score, FPR, mean_dice) = "
                f"({res['epoch']}, "
                f"{res['score']:.6f}, "
                f"{res['fpr']:.6f}, "
                f"{res['mean_dice']:.6f})"
            )
            lines.append("")  # Leerzeile als Trenner

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    if found_any:
        OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
        print(f"Fertig. Ergebnisse geschrieben nach: {OUT_TXT}")
    else:
        OUT_TXT.write_text("Keine gültigen Runs gefunden.\n", encoding="utf-8")
        print("Keine gültigen Runs gefunden.")

if __name__ == "__main__":
    main()
