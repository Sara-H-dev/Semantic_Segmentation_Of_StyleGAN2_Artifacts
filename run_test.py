#!/usr/bin/env python3
import subprocess, os
from pathlib import Path
import pandas as pd
from scripts.config_parser import Config_Parser
import logging


# --- Env ---
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


logger = logging.getLogger(__name__)

root_out = Path("./model_out/FINAL")
root_out.mkdir(parents=True, exist_ok=True)

# --- Fixed Args ---
cfg_path   = "./config_used.yaml"
py         = "python3"
train_py   = "train.py"
test_py   = "test.py"

# --- Metrik/CSV ---
CSV_NAME   = "val_metric_all_epoch.csv"  # <- konsistent bleiben
METRIC_COL = "Score"           # <- Spaltenname in der CSV

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


# ------------------------------------CONFIG------------------------------------------------------------ #


logging.info("Final Run")


out = root_out / f"final_run_1"
cfg_path = out / cfg_path


bm = out / "best_model.pth"
if not bm.exists():
    raise FileNotFoundError(f"{bm} not found")

cmd = [
    py, test_py,
    "--cfg", cfg_path,
    "--check_point_dir", str(out),
    "--out", str(root_out)
]


subprocess.run(cmd, env=env, check=True)

logging.info(f"END of Evaluation")
