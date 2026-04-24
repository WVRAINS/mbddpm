import os
import pandas as pd
from datetime import datetime
import torch


def save_samples(samples,
                 taxa_list=None,
                 data_name='data',
                 add_method="code",
                 sampling="DDPM",
                 num_epochs=0,
                 save_dir="result"):
    if isinstance(samples, torch.Tensor):

        if samples.dim() > 2:
            samples = samples.view(samples.shape[0], -1)

        samples_numpy = samples.detach().cpu().numpy()

    else:
        samples_numpy = samples

    if taxa_list is not None:
        samples_numpy = samples_numpy[:, :len(taxa_list)]
        df = pd.DataFrame(samples_numpy, columns=taxa_list)
    else:
        df = pd.DataFrame(samples_numpy)

    save_path = f'./generated/{data_name}'
    os.makedirs(save_path, exist_ok=True)

    generate_time = datetime.now().strftime("%m%d-%H%M%S")

    filename = (
        f'epoch_{num_epochs}_'
        f'{add_method}_'
        f'{sampling}_'
        f'{generate_time}.csv'
    )

    full_path = os.path.join(save_path, filename)

    df.to_csv(full_path, index=False)

    print(f"Samples saved to: {full_path}")