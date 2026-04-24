import torch
from timm.utils import ModelEmaV3

from mbddpm.models.unet import UNET
from mbddpm.models.scheduler import DDPMScheduler


class Sampler:

    def __init__(
        self,
        checkpoint_path,
        num_time_steps=1000,
        generate_num=10,
        device="cpu",
    ):

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.generate_num = generate_num
        self.num_time_steps = num_time_steps

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.data_shape = checkpoint["data_shape"]
        self.num_features = checkpoint["num_features"]
        self.taxa_list = checkpoint["taxa_list"]
        self.scheduler = DDPMScheduler(num_time_steps).to(self.device)
        self.model = UNET().to(self.device)
        self.ema = ModelEmaV3(self.model)
        self.model.load_state_dict(checkpoint["weights"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.ema.module.eval()

    @torch.no_grad()
    def sample(self, num=None):

        if num is not None:
            self.generate_num = num

        x = torch.randn(
            (self.generate_num, 1, self.data_shape[0], self.data_shape[1]),
            device=self.device
        )

        for t in reversed(range(self.num_time_steps)):

            t_tensor = torch.full(
                (self.generate_num,),
                t,
                device=self.device,
                dtype=torch.long
            )

            pred_noise = self.ema.module(x, t_tensor)

            alpha = self.scheduler.alpha[t]
            alpha_hat = self.scheduler.alpha_hat[t]
            beta = self.scheduler.beta[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise
            )
            print(1000 - t)
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta) * noise

        # flatten
        samples = x.view(x.shape[0], -1)

        # padding
        samples = samples[:, :self.num_features]

        return samples