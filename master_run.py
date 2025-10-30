#!/usr/bin/env python3
import subprocess, os
from pathlib import Path
import pandas as pd

# --- Env ---
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- Grid ---
MAX_LRS   = [2e-5, 3e-5]
weight_decay_1 = 0.0             # fester WD in Stage 1
WD_GRID   = [1e-5, 1e-4, 1e-3, 1e-2]

# --- Fixed Args ---
drop_rate  = 0.0
drop_path  = 0.05
alpha      = 0.2
cfg_path   = "./config.yaml"
py         = "python3"
train_py   = "train.py"

# --- Metrik/CSV ---
CSV_NAME   = "val_metric_fake_epoch.csv"  # <- konsistent bleiben
METRIC_COL = "mean_i_soft_dice"           # <- Spaltenname in der CSV

root_out = Path("./model_out/mega_search/weight_decay_lr1e-5_wu_20")
root_out.mkdir(parents=True, exist_ok=True)

def safe_read_csv(p: Path):
    try:
        return pd.read_csv(p, on_bad_lines="skip")
    except Exception as e:
        print(f"[WARN] Konnte {p} nicht lesen: {e}")
        return None

def get_best_from_df(df: pd.DataFrame, col_name: str):
    if df is None or col_name not in df.columns:
        return None
    s = pd.to_numeric(df[col_name], errors="coerce")
    if s.dropna().empty:
        return None
    idx = s.idxmax()
    return {"row_index": idx, "value": float(s.loc[idx])}
    
# -------- Stage 1: (LR × Warmup) bei fixem WD --------
best_val = -1.0
best_tag = None
wu = 20

"""

for lr in MAX_LRS:
        out = root_out / f"stage1_lr{lr}_wu{wu}_wd{weight_decay_1}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            py, train_py,
            "--output_dir", str(out),
            "--cfg", cfg_path,
            "--weight_decay", str(weight_decay_1),
            "--drop_path", str(drop_path),
            "--drop_rate", str(drop_rate),
            "--alpha", str(alpha),
            "--warm_up", str(wu),
            "--lr", str(lr),
        ]
        print("CMD:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
        df = safe_read_csv(out / CSV_NAME)
        res = get_best_from_df(df, METRIC_COL)

        if res is None:
            print(f"[WARN] Keine Metrik für {out}")
            continue
        val = res["value"]
        print(f"[S1] lr={lr} wu={wu} -> {METRIC_COL}={val:.4f}")
        if val > best_val:
            best_val = val
            best_tag = (lr, wu)



if best_tag is None:
    raise SystemExit("[ERROR] Stage 1 lieferte keine gültigen Ergebnisse.")

best_lr, best_wu = best_tag
print(f"[INFO] Stage 1 BEST: lr={best_lr}, warmup={best_wu}, {METRIC_COL}={best_val:.4f}")

# -------- Stage 2: WD-Sweep mit bestem (LR, Warmup) --------
best2_val = -1.0
best2_wd  = 20
"""
best_wu = 20
best_lr = 1e-5

for wd in WD_GRID:
    out = root_out / f"stage2_lr{best_lr}_wu{best_wu}_wd{wd}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        py, train_py,
        "--output_dir", str(out),
        "--cfg", cfg_path,
        "--weight_decay", str(wd),
        "--drop_path", str(drop_path),
        "--drop_rate", str(drop_rate),
        "--alpha", str(alpha),
        "--warm_up", str(best_wu),
        "--lr", str(best_lr),
    ]
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    df = safe_read_csv(out / CSV_NAME)
    res = get_best_from_df(df, METRIC_COL)
    if res is None:
        print(f"[WARN] Keine Metrik für {out}")
        continue
    val = res["value"]
    print(f"[S2] wd={wd} -> {METRIC_COL}={val:.4f}")
    if val > best2_val:
        best2_val = val
        best2_wd = wd

if best2_wd is None:
    raise SystemExit("[ERROR] Stage 2 lieferte keine gültigen Ergebnisse.")

print(f"[RESULT] BEST COMBO: lr={best_lr}, warmup={best_wu}, wd={best2_wd} -> {METRIC_COL}={best2_val:.4f}")
