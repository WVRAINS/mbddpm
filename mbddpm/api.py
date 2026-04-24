from mbddpm.training.trainer import Trainer
from mbddpm.training.sampler import Sampler


def train_model(
        data,
        taxa_list,
        data_name,
        **kwargs
):
    trainer = Trainer(
        data=data,
        taxa_list=taxa_list,
        data_name=data_name,
        **kwargs
    )

    trainer.train()

def generate_samples(checkpoint_path, **kwargs):
    sampler = Sampler(
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return sampler.sample()