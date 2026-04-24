import os

from mbddpm.data.csv_dataset import csv_dataset
from mbddpm.training.trainer import Trainer
from mbddpm.training.sampler import Sampler


def main():

    # ========= 配置 =========
    data_path = "D:\common\desktop\mbddpm\data\demo_case_first10.csv"
    device = "cpu"   # debug 一定用 cpu
    data_name = "debug_run"

    # ========= 1. 加载数据 =========
    dataset = csv_dataset(data_path)

    data = dataset.data
    taxa_list = dataset.taxa_list

    print("\n===== DATA =====")
    print("shape:", data.shape)
    print("dtype:", data.dtype)
    print("first row:", data[0])

    # 👉 这里可以打断点看 data

    # ========= 2. 训练（极小规模） =========
    trainer = Trainer(
        data=dataset,
        taxa_list=taxa_list,
        data_name=data_name,

        batch_size=4,
        num_time_steps=10,   # 小一点方便 debug
        add_method="code",

        num_epochs=2,        # 只跑2轮
        lr=1e-4,
        ema_decay=0.9,
        save_epoch=1,

        device=device
    )

    print("\n===== START TRAIN =====")
    trainer.train()

    # 👉 这里可以打断点看模型 / loss

    # ========= 3. 找 checkpoint =========
    ckpt_path = os.path.join(
        "runs",
        data_name,
        "checkpoints",
        "epoch_2_code.pt"
    )

    print("\nCheckpoint:", ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint not found!"

    # ========= 4. 采样 =========
    sampler = Sampler(
        checkpoint_path=ckpt_path,
        generate_num=5,
        device=device
    )

    print("\n===== START SAMPLING =====")
    samples = sampler.sample()

    print("\n===== SAMPLES =====")
    print("shape:", samples.shape)
    print(samples)

    # 👉 这里可以打断点看生成数据

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()