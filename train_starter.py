import subprocess
import train
import os

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


subprocess.run([
    "python3", "train.py",
    "--cfg", "./config.yaml"],
env=env,
check=True, # catches errors
)