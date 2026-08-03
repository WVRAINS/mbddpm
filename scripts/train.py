import argparse
from mbddpm.api import train_model
from mbddpm.utils.seed import set_seed
import yaml

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="../configs/config_IBD_case.yaml")

    args = parser.parse_args()

    # 读取配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设置随机种子
    set_seed(cfg["experiment"]["seed"])

    # 训练
    train_model(
        data=cfg["experiment"]["dataset"],
        data_name=cfg["experiment"]["name"],
        device=cfg["device"],
        batch_size=cfg["data"]["batch_size"],
        num_epochs=cfg["training"]["num_epochs"],
        save_epoch=cfg["training"]["save_epoch"],
        num_time_steps=cfg["model"]["num_time_steps"],
        lr=cfg["training"]["lr"],
        ema_decay=cfg["training"]["ema_decay"],
    )

if __name__ == "__main__":
    main()