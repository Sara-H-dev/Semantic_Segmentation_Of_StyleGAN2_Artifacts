import subprocess
import train
import os

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

wd = 0.1
drop_rate = 0.0
drop_path = 0.05
alpha = 0.2
out_dir = f"./model_out/SymmetricUnifiedFocalLoss"

subprocess.run([
    "python3", "train.py",
        "--output_dir", out_dir,
        "--weight_decay", str(wd),                 
        "--drop_path", str(drop_path),  
        "--drop_rate", str(drop_rate),  
        "--alpha", str(alpha),
        "--cfg", "./config.yaml",
    ],
env=env,
check=True, # catches errors
)
