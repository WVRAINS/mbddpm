import os
import sys
import platform
from datetime import datetime
from pathlib import Path

from mbddpm.utils.get_project_root import get_project_root
import pandas as pd
import torch


def get_gpu_info(device="cuda:0"):

    info = {
        "gpu_name": "CPU",
        "gpu_count": 0,
        "gpu_memory_GB": 0,
        "cuda_version": "None"
    }

    if torch.cuda.is_available():
        try:
            gpu_id = int(str(device).split(":")[-1])
        except:
            gpu_id = 0

        props = torch.cuda.get_device_properties(gpu_id)
        info["gpu_name"] = (props.name)
        info["gpu_count"] = (torch.cuda.device_count())
        info["gpu_memory_GB"] = round(props.total_memory / (1024 ** 3), 2)
        info["cuda_version"] = (torch.version.cuda)
    return info

def get_system_info(device="cuda:0"):

    info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "system": platform.system(),
        "system_version": platform.version(),
        "machine": platform.machine(),
    }
    info.update(get_gpu_info(device))
    return info

def get_checkpoint_size(path):
    if os.path.exists(path):
        size = (os.path.getsize(path)/(1024 ** 2))
        return round(size, 3)
    return 0

def append_excel_record(record):

    record_path = (get_project_root() / "generated" / "experiment_record.xlsx")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([record])

    if record_path.exists():
        df_old = pd.read_excel(record_path)
        df = pd.concat([df_old,df_new],ignore_index=True)
    else:
        df = df_new
    with pd.ExcelWriter(record_path,engine="openpyxl") as writer:
        df.to_excel(writer,index=False,sheet_name="experiment")
        worksheet = writer.sheets["experiment"]

        for column in worksheet.columns:
            max_length = 0
            column_letter = (column[0].column_letter)
            for cell in column:
                try:
                    length = len(str(cell.value))
                    if length > max_length:
                        max_length = length
                except:
                    pass
            worksheet.column_dimensions[column_letter].width = max_length + 3

    print("=" * 70)
    print("Experiment record saved:")
    print(record_path)
    print("=" * 70)

def save_experiment_record(
        *,
        experiment_id,
        data_name,
        checkpoint_path,
        epoch,
        train_start_time,
        checkpoint_time,
        data_shape,
        batch_size,
        num_epochs,
        lr,
        ema_decay,
        num_time_steps,
        device,
        model_name,
        last_loss=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_size = get_checkpoint_size(checkpoint_path)
    num_samples = (data_shape[0] if data_shape else None)
    num_features = (data_shape[1] if data_shape else None)
    elapsed_seconds = (checkpoint_time - train_start_time).total_seconds()
    record = {
        "experiment_id": experiment_id,
        "data_name": data_name,
        "model": model_name,
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "epoch": epoch,
        "train_start_time": train_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_time": checkpoint_time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed_seconds,3),
        "num_samples": num_samples,
        "num_features": num_features,
        "data_shape": str(data_shape),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "ema_decay": ema_decay,
        "num_time_steps": num_time_steps,
        "last_loss": last_loss,
        "checkpoint_size_MB": checkpoint_size,
        "device": str(device)
    }

    record.update(get_system_info(device))

    append_excel_record(record)