import argparse
import torch
from mbddpm.api import generate_samples
from mbddpm.utils.save_sample import save_samples
import yaml

from mbddpm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="../configs/config.yaml")

    args = parser.parse_args()
    # get config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])


    samples = generate_samples(
        checkpoint = cfg["sampling"]["checkpoint"],
        device = cfg["device"],
        generate_num = cfg["sampling"]["generate_num"],
    )

    checkpoint = torch.load(cfg["sampling"]["checkpoint"], cfg["device"])

    save_samples(
        samples,
        taxa_list = checkpoint["taxa_list"],
        data_name = checkpoint["data_name"],
        num_epochs = checkpoint["epoch"]
    )

if __name__ == "__main__":
    main()