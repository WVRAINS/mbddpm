import pandas as pd
import torch
from torch.utils.data import Dataset
from mbddpm.utils.process_data import process_data


class csv_dataset(Dataset):

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        if not pd.api.types.is_numeric_dtype(df.iloc[:,0]):
            df = df.iloc[:,1:]
        matrix = df.values.astype("float32")
        tensor, shape = process_data(matrix, "code")

        self.data = torch.tensor(tensor)
        self.shape = shape
        self.num_features = matrix.shape[1]
        self.taxa_list = df.columns.tolist()

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]