import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import ModelEmaV3

from mbddpm.models.unet import UNET
from mbddpm.models.scheduler import DDPMScheduler

class Trainer:
    def __init__(
        self,
        data,
        taxa_list,
        data_name,
        batch_size=16,
        num_time_steps=1000,
        lr=1e-5,
        ema_decay=0.9999,
        add_method='code',
        device='cpu',
        num_epochs=10000,
        save_epoch=10000,
        checkpoint_path=None,
    ):
        self.data = data
        self.taxa_list = taxa_list
        self.data_name = data_name
        self.num_features = len(taxa_list)
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

    def train(self):
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
        save_dir = os.path.join("runs", self.data_name)
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(
            save_dir,
            f"epoch_{epoch}_{self.add_method}.pt"
        )

        checkpoint = {
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "data_shape": [self.data.shape[0], self.data.shape[1]],
            "num_features": self.num_features,
            "taxa_list": self.taxa_list
        }

        torch.save(checkpoint, path)
        print("Saved:", path)