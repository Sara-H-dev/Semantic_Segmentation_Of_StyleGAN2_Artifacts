import subprocess, os

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

for wd in [0, 1e-1 , 1e-2]:
    cmd = [
        "python3", "train.py",
        "--output_dir", f"./model_out/weight_decay_search_211025_wd_{wd}",
        "--weight_decay", str(wd),                 # ← Flag + Wert
        "--cfg", "./config.yaml",
    ]
    print("CMD:", " ".join(cmd))                  # Debug
    subprocess.run(cmd, env=env, check=True)