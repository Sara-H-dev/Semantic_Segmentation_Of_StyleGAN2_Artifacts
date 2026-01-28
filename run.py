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

# =========== IMPORTANT PATHS ========================
# outputpath of the results:
root_out = Path("./model_out/RUN1")
root_out.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename='./model_out/RUN1/run.log', encoding='utf-8', level=logging.DEBUG)

# path to the hyperparameters file:
cfg_path   = "./config.yaml"
py         = "python3"
train_py   = "train.py"

# --- Metrik/CSV ---
CSV_NAME   = "val_metric_all_epoch.csv"  # this is the name of the file there the metrics of each epoch are stored
METRIC_COL = "Score"                     # this ist the colume that the script uses to determin the best model
                                         # you can change it, for example for the best dice score or whatever      

# ======== Stuff for parsing the yaml ================ #
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

# ============== HYPERPARAMETER ========
# Here you can set the Hyperparameter like you want, and you can do a gridsearch

ATTN_DROP = [0.1]
ALPHA = [0.3, 0.4]
LEARNING_RATE = [8.5e-6, 3e-5]

drop_rate_0 = 0.0
attn_drop_0 = 0.0
drop_path_0 = 0.05

# ------------------------------------DROP RATE------------------------------------------------------------ #
wd = 0.001
best_drop_path = 0.1
best_drop_rate = 0.0

# ------------------------------------ATTN DROP------------------------------------------------------------- #

result_dict_att = {}
result_list_att = []
result_path_dict = {}
res_dict = {}

logging.info("Attention drop search:")

for attn_drop in ATTN_DROP:
    out = root_out / f"drop_path{best_drop_path:.2f}_drop_rate{best_drop_rate:.2f}_attn_drop{attn_drop:.2f}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", attn_drop)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", 0.2)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", 0.8)
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
    result_dict_att[score] = attn_drop
    result_list_att.append(score)
    result_path_dict[attn_drop] = out
    logging.info(f"drop_path{best_drop_path}_drop_rate{best_drop_rate}_attn_drop{attn_drop}: result {score}")

best_att_score = max(result_list_att)
best_att = result_dict_att[best_att_score]
best_path = result_path_dict[best_att]

logging.info(f"Best model with attention drop {best_att} in {best_path}")

# ------------------------------------Alpha Refine------------------------------------------------------------- #
result_dict_alpha = {}
result_list_alpha = []
result_path_dict_alpha = {}
res_dict = {}

logging.info("Alpha refine:")

for alpha in ALPHA:
    beta = 1 - alpha
    out = root_out / f"alpha_{alpha:.2f}_drop_path{best_drop_path:.2f}_drop_rate{best_drop_rate:.2f}_attn_drop{best_att:.2f}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", best_att)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", alpha)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", beta)
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
    result_dict_alpha[score] = alpha
    result_list_alpha.append(score)
    result_path_dict_alpha[alpha] = out
    logging.info(f"alpha_{alpha}_drop_path{best_drop_path}_drop_rate{best_drop_rate}_attn_drop{best_att}: result{score}")

best_alpha_score = max(result_list_alpha)
best_alpha = result_dict_alpha[best_alpha_score]
best_path = result_path_dict_alpha[best_alpha]
best_beta = 1 - best_alpha

logging.info(f"Best model with alpha {best_alpha} in {best_path}")

# ------------------------------------LEARNING RATE REFINE !!------------------------------------------------------------- #


result_dict_lr = {}
result_list_lr = []
result_path_dict = {}
res_dict = {}
result_dict_lr[best_alpha_score] = 1e-5
result_list_lr.append(best_alpha_score)
result_path_dict[1e-5] = best_path

logging.info("Learning Rate search:")

for lr in LEARNING_RATE:
    out = root_out / f"lr_{lr}_drop_path{best_drop_path:.2f}_drop_rate{best_drop_rate:.2f}_attn_drop{best_att:.2f}"
    out.mkdir(parents=True, exist_ok=True)
    config_parser.set_yaml_value("OUTPUT_DIR", str(out))
    config_parser.set_yaml_value("TRAIN.WEIGHT_DECAY", wd)
    config_parser.set_yaml_value("MODEL.DROP_RATE", best_drop_rate)
    config_parser.set_yaml_value("MODEL.DROP_PATH_RATE", best_drop_path)
    config_parser.set_yaml_value("MODEL.ATTN_DROP_RATE", best_att)
    config_parser.set_yaml_value("TRAIN.BASE_LR", lr)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_ALPHA", best_alpha)
    config_parser.set_yaml_value("TRAIN.TVERSKY_LOSS_BETA", best_beta)
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
    result_dict_lr[score] = lr
    result_list_lr.append(score)
    result_path_dict[lr] = out
    logging.info(f"lr_{lr}_drop_path{best_drop_path}_drop_rate{best_drop_rate}_attn_drop{best_att}: result{score}")

best_lr_score = max(result_list_lr)
best_lr = result_dict_lr[best_lr_score]

logging.info(f"Best model with lr {best_lr} is the best model!!")