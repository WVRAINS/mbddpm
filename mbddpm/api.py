import torch
from mbddpm.training.trainer import Trainer
from mbddpm.training.sampler import Sampler


def train_model(
        data,
        data_name,
        **kwargs
):
    trainer = Trainer(
        data=data,
        data_name=data_name,
        **kwargs
    )

    trainer.train()


def generate_samples(checkpoint_path, generate_num, device, **kwargs):
    checkpoint = torch.load(checkpoint_path, device)
    sampler = Sampler(
        checkpoint=checkpoint,
        generate_num=generate_num,
        device=device,
        **kwargs
    )
    return sampler.sample()