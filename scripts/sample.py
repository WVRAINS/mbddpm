import argparse
from mbddpm.api import generate_samples
from mbddpm.utils.save_sample import save_samples
from mbddpm.data.csv_dataset import csv_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    dataset = csv_dataset(args.data_path)

    samples = generate_samples(
        checkpoint_path=args.checkpoint,
        device=args.device,
        generate_num=args.num,
    )

    save_samples(
        samples,
        taxa_list=dataset.taxa_list,
        data_name="generated",
        num_epochs=0
    )

if __name__ == "__main__":
    main()