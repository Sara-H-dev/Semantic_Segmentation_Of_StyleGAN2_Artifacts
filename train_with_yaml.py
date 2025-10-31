#!/usr/bin/env python3
import subprocess, os
from pathlib import Path
import pandas as pd
from scripts.config_parser import Config_Parser

# --- Env ---
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- Grid ---
MAX_LRS   = [2e-5, 3e-5]
weight_decay_1 = 0.0             # fester WD in Stage 1
WD_GRID   = [1e-4]

# --- Fixed Args ---
cfg_path   = "./config.yaml"
py         = "python3"
train_py   = "train.py"

# --- Metrik/CSV ---
CSV_NAME   = "val_metric_fake_epoch.csv"  # <- konsistent bleiben
METRIC_COL = "mean_i_soft_dice"           # <- Spaltenname in der CSV


root_out = Path("./model_out/mega_search/weight_decay_lr1e-5_wu_20")
root_out.mkdir(parents=True, exist_ok=True)
config_parser = Config_Parser(yaml_path= cfg_path, create_missing= False, preserve_formatting= False)

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
    
best_wu = 20
best_lr = 1e-5

for wd in WD_GRID:
    out = root_out / f"stage2_lr{best_lr}_wu{best_wu}_wd{wd}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", 0.0)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", 0.05)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", 0.0)
    cmd = [
        py, train_py,
        "--cfg", cfg_path,
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


