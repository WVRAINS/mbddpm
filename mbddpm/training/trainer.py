import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import ModelEmaV3
from datetime import datetime

from mbddpm.utils.csv_dataset import csv_dataset
from mbddpm.utils.experiment_logger import save_experiment_record, append_excel_record, get_checkpoint_size, \
    get_system_info
from mbddpm.models.unet import UNET
from mbddpm.models.scheduler import DDPMScheduler
from mbddpm.utils.get_project_root import get_project_root
from mbddpm.utils.seed import set_seed


class Trainer:
    def __init__(
        self,
        data,
        data_name,
        seed=42,
        batch_size=16,
        num_time_steps=1000,
        lr=1e-5,
        ema_decay=0.9999,
        add_method='code',
        device='cpu',
        num_epochs=150000,
        save_epoch=150000,
        checkpoint_path=None,
    ):
        dataset = csv_dataset(data)
        self.data = dataset.data
        self.taxa_list = dataset.taxa_list
        self.data_name = data_name
        self.seed = seed
        self.num_features = len(self.taxa_list)
        self.batch_size = batch_size
        self.num_time_steps = num_time_steps
        self.add_method = add_method
        self.num_epochs = num_epochs
        self.save_epoch = save_epoch

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)
        self.loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.scheduler = DDPMScheduler(num_time_steps).to(self.device)
        self.model = UNET().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))
        self.ema = ModelEmaV3(self.model, decay=ema_decay)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.criterion = nn.MSELoss()
        # time
        self.train_start_time = None
        self.current_loss = None
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(self):
        set_seed(self.seed)
        # 记录训练开始时间
        self.train_start_time = datetime.now()
        for epoch in range(self.num_epochs):
            self.model.train()
            with tqdm(self.loader, desc=f"Epoch [{epoch+1}/{self.num_epochs}]") as bar:
                for x in bar:
                    x = x.float().to(self.device)
                    b = x.shape[0]
                    t = torch.randint(0, self.num_time_steps, (b,), device=self.device)
                    noise = torch.randn_like(x)
                    if self.add_method == 'code':
                        alpha_hat = self.scheduler.alpha_hat[t].view(b, 1, 1, 1)
                    else:
                        alpha_hat = self.scheduler.alpha_hat[t].view(b, 1, 1)
                    x_noisy = torch.sqrt(alpha_hat) * x + torch.sqrt(1 - alpha_hat) * noise
                    pred = self.model(x_noisy, t)
                    loss = self.criterion(pred, noise)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.ema.update(self.model)
                    bar.set_postfix(loss=float(loss.item()))
            if (epoch + 1) % self.save_epoch == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch):
        checkpoint_dir = (get_project_root() / "checkpoint" / self.data_name)
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        path = checkpoint_dir / f"epoch_{epoch}_"f"{self.data_name}_"f"{self.add_method}.pt"

        checkpoint = {
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "data_shape": [self.data.shape[2], self.data.shape[3]],
            "num_features": self.num_features,
            "taxa_list": self.taxa_list,
            "data_name": self.data_name,
            "num_time_steps": self.num_time_steps,
            "epoch": epoch,
        }

        torch.save(checkpoint, path)
        checkpoint_time = datetime.now()
        print("Saved:", path)

        checkpoint_size = get_checkpoint_size(path)
        elapsed_seconds = (checkpoint_time - self.train_start_time).total_seconds()
        record = {
            "experiment_id": self.experiment_id,
            "event_type": "training",
            "data_name": self.data_name,
            "model": "MB-DDPM",
            "checkpoint": path.name,
            "checkpoint_path": str(path.resolve()),
            "epoch": epoch,
            "train_start_time": self.train_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_time": checkpoint_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "num_samples": self.data.shape[0],
            "num_features": self.num_features,
            "data_shape": str(self.data.shape),
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "ema_decay": getattr(self.ema, "decay", None),
            "num_time_steps": self.num_time_steps,
            "last_loss": self.current_loss,
            "checkpoint_size_MB": checkpoint_size,
            "device": str(self.device)
        }
        record.update(get_system_info(self.device))

        append_excel_record(record)