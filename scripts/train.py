import argparse
from mbddpm.api import train_model
from mbddpm.data.csv_dataset import csv_dataset
from mbddpm.utils.seed import set_seed
import yaml

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # 读取配置
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 设置随机种子
    set_seed(cfg["experiment"]["seed"])

    # 加载数据
    dataset = csv_dataset(args.data_path)

    # 训练
    train_model(
        data=dataset.data,
        taxa_list=dataset.taxa_list,
        data_name=cfg["experiment"]["name"],
        device=args.device,
        batch_size=cfg["data"]["batch_size"],
        num_epochs=cfg["training"]["num_epochs"],
        save_epoch=cfg["training"]["save_epoch"],
        num_time_steps=cfg["model"]["num_time_steps"],
        lr=cfg["training"]["lr"],
        ema_decay=cfg["training"]["ema_decay"],
    )

if __name__ == "__main__":
    main()