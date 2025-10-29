import subprocess, os, csv

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

wd = 0.1
drop_rate = 0.0
drop_path = 0.05

for alpha in [0.35, 0.3 , 0.25, 0.2, 0.15, 0.1]:
    beta = 1 - alpha
    out_dir = f"./model_out/parameter_search/alpha_{alpha}"
    cmd = [
        "python3", "train.py",
        "--output_dir", out_dir,
        "--weight_decay", str(wd),                 
        "--drop_path", str(drop_path),  
        "--drop_rate", str(drop_rate),  
        "--alpha", str(alpha),
        "--cfg", "./config.yaml",
    ]
    print("CMD:", " ".join(cmd))                  # Debug
    subprocess.run(cmd, env=env, check=True)
