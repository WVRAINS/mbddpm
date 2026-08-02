import argparse
import os
import torch
from mbddpm.utils.config import load_config
from mbddpm.utils.seed import set_seed
from mbddpm.utils.save_sample import save_samples
from mbddpm.training.trainer import Trainer
from mbddpm.training.sampler import Sampler


# ======================
# Train
# ======================
def train_cmd(args):

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)

    set_seed(cfg["experiment"]["seed"])

    trainer = Trainer(
        data=cfg["experiment"]["dataset"],
        data_name=cfg["experiment"]["name"],
        seed=cfg["experiment"]["seed"],
        batch_size=cfg["data"]["batch_size"],
        num_time_steps=cfg["model"]["num_time_steps"],
        add_method=cfg["model"]["add_method"],
        num_epochs=cfg["training"]["num_epochs"],
        lr=cfg["training"]["lr"],
        ema_decay=cfg["training"]["ema_decay"],
        save_epoch=cfg["training"]["save_epoch"],
        device=cfg["device"],
    )

    trainer.train()


# ======================
# Sample
# ======================
def sample_cmd(args):

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)

    set_seed(cfg["experiment"]["seed"])

    checkpoint = torch.load(cfg["sampling"]["checkpoint"], cfg["device"])

    sampler = Sampler(
        checkpoint=checkpoint,
        generate_num=cfg["sampling"]["generate_num"],
        device=cfg["device"]
    )

    samples = sampler.sample()

    save_samples(
        samples,
        taxa_list=checkpoint.taxa_list,
        data_name=checkpoint.data_name,
        num_epochs=checkpoint.epoch
    )

# ======================
# Main CLI
# ======================
def main():
    parser = argparse.ArgumentParser("mbddpm")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("config", type=str, help="Provide the path to the YAML configuration.")

    # sample
    sample_parser = subparsers.add_parser("sample", help="Generate samples")
    sample_parser.add_argument("config", type=str, help="Provide the path to the YAML configuration.")

    args = parser.parse_args()

    if args.command == "train":
        train_cmd(args)
    elif args.command == "sample":
        sample_cmd(args)


if __name__ == "__main__":
    main()