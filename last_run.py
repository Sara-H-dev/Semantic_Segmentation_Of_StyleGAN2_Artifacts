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
logging.basicConfig(filename='./model_out/FINAL/final.log', encoding='utf-8', level=logging.DEBUG)

# --- Fixed Args ---
cfg_path   = "./config.yaml"
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
wd = 0.001
best_drop_path = 0.1
best_drop_rate = 0.0
attn_drop = 0.0
seed = 53
mix = 0.5
pretraining = "segface" #"imagenet1k"
epoch = 100
# ------------------------------------ FINAL RUN ------------------------------------------------------------- #

res_dict = {}


logging.info("Final Run")


out = root_out / f"final_run_1"
out.mkdir(parents=True, exist_ok=True)
config_parser.set_yaml_value("OUTPUT_DIR", str(out))
config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", attn_drop)
config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", 0.2)
config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", 0.8)
config_parser.set_yaml_value("SEED", seed)
config_parser.set_yaml_value("TRAIN.LOSS_TVERSKY_BCE_MIX", mix)
config_parser.set_yaml_value("MODEL.PRETRAIN_WEIGHTS", pretraining)
config_parser.set_yaml_value("TRAIN.MAX_EPOCHS", epoch)
config_parser.set_yaml_value("SAVE_BEST_RUN", True)
config_parser.set_yaml_value("SHOW_PREDICTION", 167)

cmd = [
    py, train_py,
    "--cfg", cfg_path,
]

print("CMD:", " ".join(cmd))
subprocess.run(cmd, env=env, check=True)
df = safe_read_csv(out / CSV_NAME)
res_dict = get_best_from_df(df, METRIC_COL)
if res_dict is None:
        raise ValueError("res dictionary is empty")
score = res_dict["value"]

logging.info(f"final_run_ with_score_{score}")

logging.info(f"Starting evaluation with test set")

bm = out / "best_model.pth"
if not bm.exists():
    raise FileNotFoundError(f"{bm} not found")

cmd = [
    py, test_py,
    "--cfg", cfg_path,
    "--check_point_dir", str(out)
]

print("CMD:", " ".join(cmd))
subprocess.run(cmd, env=env, check=True)

logging.info(f"END of Evaluation")

