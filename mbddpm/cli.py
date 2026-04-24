import argparse
import os

from mbddpm.data.csv_dataset import csv_dataset
from mbddpm.utils.config import load_config
from mbddpm.utils.device import get_device
from mbddpm.utils.seed import set_seed
from mbddpm.utils.save_sample import save_samples
from mbddpm.training.trainer import Trainer
from mbddpm.training.sampler import Sampler


# ======================
# Train
# ======================
def train_cmd(args):

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)

    device = get_device(None if args.device == "auto" else args.device)

    set_seed(args.seed)

    dataset = csv_dataset(args.data)

    trainer = Trainer(
        data=dataset,
        taxa_list=dataset.taxa_list,
        data_name=cfg["experiment"]["name"],

        batch_size=cfg["data"]["batch_size"],

        num_time_steps=cfg["model"]["num_time_steps"],
        add_method=cfg["model"]["add_method"],

        num_epochs=cfg["training"]["num_epochs"],
        lr=cfg["training"]["lr"],
        ema_decay=cfg["training"]["ema_decay"],
        save_epoch=cfg["training"]["save_epoch"],

        device=device
    )

    trainer.train()


# ======================
# Sample
# ======================
def sample_cmd(args):
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)
    data_name = cfg["experiment"]["name"],

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = get_device(None if args.device == "auto" else args.device)

    sampler = Sampler(
        checkpoint_path=args.checkpoint,
        generate_num=args.num,
        device=device
    )

    samples = sampler.sample()

    save_samples(
        samples,
        taxa_list=sampler.taxa_list,
        data_name=data_name,
        num_epochs=args.num
    )


# ======================
# Main CLI
# ======================
def main():
    parser = argparse.ArgumentParser("mbddpm")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("data", type=str, help="CSV dataset")
    train_parser.add_argument("config", type=str, help="config yaml")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=42)

    # sample
    sample_parser = subparsers.add_parser("sample", help="Generate samples")
    sample_parser.add_argument("checkpoint", type=str)
    sample_parser.add_argument("config", type=str, help="config yaml")
    sample_parser.add_argument("--num", type=int, default=10)
    sample_parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.command == "train":
        train_cmd(args)
    elif args.command == "sample":
        sample_cmd(args)


if __name__ == "__main__":
    main()