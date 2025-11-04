#!/usr/bin/env python3
import subprocess, os
from pathlib import Path
import pandas as pd
from scripts.config_parser import Config_Parser
import logging


# --- Env ---
env = os.environ.copy()
#env["CUDA_VISIBLE_DEVICES"] = "0"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
#env["CUDNN_BENCHMARK"] = "0"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- Grid ---

SEED = [120, 456, 132]


logger = logging.getLogger(__name__)

root_out = Path("./model_out/SEED")
root_out.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename='./model_out/SEED/seed_run.log', encoding='utf-8', level=logging.DEBUG)

# --- Fixed Args ---
cfg_path   = "./config.yaml"
py         = "python3"
train_py   = "train.py"

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



# ------------------------------------DROP RATE------------------------------------------------------------ #
wd = 0.001
best_drop_path = 0.1
best_drop_rate = 0.0

# ------------------------------------ ATTN DROP 0.05 ------------------------------------------------------------- #

result_dict_att_005 = {}
result_list_att_005 = []
res_dict = {}
result_dict_att_005[0.531960] = 1234
result_list_att_005.append(0.531960)

logging.info("Seed search for attention 0.05 drop search:")

for seed in SEED:
    out = root_out / f"attn_drop{0.05}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", 0.05)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", 0.2)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", 0.8)
    config_parser.set_yaml_value("SEED", seed)
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
    result_dict_att_005[score] = seed
    result_list_att_005.append(score)
    logging.info(f"attn_drop{0.05} seed {seed}: result {score}")

average_score_005 = sum(result_list_att_005) / len(result_list_att_005)
logging.info(f"Avg_score for 0.05: {average_score_005}")


# ------------------------------------ATTN DROP 0------------------------------------------------------------- #

result_dict_att_0 = {}
result_list_att_0 = []
res_dict = {}
result_dict_att_0[0.544624] = 1234
result_list_att_0.append(0.544624)


logging.info("Seed search for attention 0.0 drop search:")

for seed in SEED:
    out = root_out / f"attn_drop{0.0}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", 0.0)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", 0.2)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", 0.8)
    config_parser.set_yaml_value("SEED", seed)
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
    result_dict_att_0[score] = seed
    result_list_att_0.append(score)
    logging.info(f"attn_drop{0.0} seed {seed}: result {score}")

average_score_0 = sum(result_list_att_0) / len(result_list_att_0)
logging.info(f"Avg_score for 0.00: {average_score_0}")

# ------------------------------------ SegFace ------------------------------------------------------------- #

result_dict = {}
result_list = []
res_dict = {}


if average_score_0 > average_score_005:
    best = average_score_0
    attn = 0.0
else:
    attn = 0.05
    best = average_score_005

for seed in SEED:
    out = root_out / f"SegFace_attn_drop{attn}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", attn)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", 0.2)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", 0.8)
    config_parser.set_yaml_value("SEED", seed)
    config_parser.set_yaml_value("MODEL.PRETRAIN_WEIGHTS", "segface")
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
    result_dict[score] = seed
    result_list.append(score)
    logging.info(f"attn_drop{attn} seed {seed}: result {score}")

average_score_SegFACE = sum(result_list) / len(result_list)
logging.info(f"segface_attn_drop{attn} seed {seed}: average result {average_score_SegFACE}")

if average_score_SegFACE > best:
    pretraining = "segface"
    best = average_score_SegFACE
else:
    pretraining = "imagenet1k"

# ------------------------------------ MIX Parameter ------------------------------------------------------------- #

MIX = [0.6, 0.7, 0.8]

result_dict_mix = {}
result_list_mix = []
res_dict = {}
result_dict_mix[best] = 0.5
result_list_mix.append(best)
seed = 1234


for mix in MIX:
    out = root_out / f"mix{mix}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", attn)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", 0.2)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", 0.8)
    config_parser.set_yaml_value("TRAIN.LOSS_TVERSKY_BCE_MIX", mix)
    config_parser.set_yaml_value("SEED", seed)
    config_parser.set_yaml_value("MODEL.PRETRAIN_WEIGHTS", pretraining)
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
    result_dict_mix[score] = mix
    result_list_mix.append(score)
    logging.info(f"mix {mix}: result {score}")

best_mix_score = max(result_list_mix)
best_mix = result_dict_mix[best_mix_score]

logging.info(f"best mix is: {best_mix}")

