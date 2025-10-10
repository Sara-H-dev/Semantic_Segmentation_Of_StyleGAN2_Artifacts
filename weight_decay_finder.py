import subprocess
import train
import os

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

for wd in [1e-2, 1e-1]:
    subprocess.run([
        "python3", "train.py",
        "--weight_decay", str(wd),
        "--cfg", "./config.yaml"],
    env=env,
    check=True, # catches errors
    )