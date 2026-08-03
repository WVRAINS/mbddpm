import yaml
import os
import shutil
from datetime import datetime

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def create_experiment_dir(cfg):
    name = cfg["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = os.path.join("runs", f"{name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    return exp_dir


def save_config_copy(cfg_path, exp_dir):
    shutil.copy(cfg_path, os.path.join(exp_dir, "config_IBD_case.yaml"))