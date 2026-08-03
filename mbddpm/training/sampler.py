import torch
from timm.utils import ModelEmaV3
from datetime import datetime
from tqdm import tqdm
from mbddpm.utils.experiment_logger import append_excel_record, get_system_info
from mbddpm.models.unet import UNET
from mbddpm.models.scheduler import DDPMScheduler


class Sampler:

    def __init__(
        self,
        checkpoint,
        generate_num=1000,
        device="cpu",
    ):
        self.checkpoint = checkpoint
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.data_shape = checkpoint["data_shape"]
        self.num_time_steps = checkpoint["num_time_steps"]
        self.generate_num = generate_num
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_features = checkpoint["num_features"]
        self.taxa_list = checkpoint["taxa_list"]
        self.scheduler = DDPMScheduler(checkpoint["num_time_steps"]).to(self.device)
        self.model = UNET().to(self.device)
        self.ema = ModelEmaV3(self.model)
        self.model.load_state_dict(checkpoint["weights"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.ema.module.eval()

    @torch.no_grad()
    def sample(self):
        sample_start_time = datetime.now()

        x = torch.randn((self.generate_num, 1, self.data_shape[0], self.data_shape[1]),device=self.device)

        for t in tqdm(
                reversed(range(self.num_time_steps)),
                total=self.num_time_steps,
                desc="Sampling",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}]"
        ):
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
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta) * noise

        # flatten
        samples = x.view(x.shape[0], -1)

        # padding
        samples = samples[:, :self.num_features]
        # end sampling
        sample_end_time = datetime.now()
        elapsed_seconds = (sample_end_time-sample_start_time).total_seconds()
        record = {
            "experiment_id":self.experiment_id,
            "event_type": "sampling",
            "generate_num":self.generate_num,
            "num_time_steps":self.num_time_steps,
            "num_features":self.num_features,
            "data_shape":str(self.data_shape),
            "sample_start_time":sample_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_end_time":sample_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sampling_seconds":round(elapsed_seconds,3),
            "device":str(self.device)
        }
        # GPU
        record.update(get_system_info(self.device))

        append_excel_record(record)

        return samples