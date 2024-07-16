import torch
from torch.utils.data import Dataset
import pandas as pd

class CSVDataset(Dataset):
    # def __init__(self, clinical_file, tumor_dl_file, fat_dl_file,):
    def __init__(self, tumor_dl_file, fat_dl_file, ):
        # self.clinical_data = pd.read_csv(clinical_file)
        # self.clinical_features = self.clinical_data.iloc[:, 2:].values

        self.tumor_dl_data = pd.read_csv(tumor_dl_file)
        self.tumor_dl_features = self.tumor_dl_data.iloc[:, 2:].values
        self.labels = self.tumor_dl_data.iloc[:, 1].values

        self.fat_dl_data = pd.read_csv(fat_dl_file)
        self.fat_dl_features = self.fat_dl_data.iloc[:, 2:].values

    def __len__(self):
        return len(self.tumor_dl_data)

    def __getitem__(self, idx):

        # clinical_feature = torch.tensor(self.clinical_features[idx], dtype=torch.float64)
        # clinical_feature = torch.unsqueeze(clinical_feature, dim=0)

        tumor_dl_feature = torch.tensor(self.tumor_dl_features[idx], dtype=torch.float64)
        tumor_dl_feature = torch.unsqueeze(tumor_dl_feature, dim=0)

        fat_dl_feature = torch.tensor(self.fat_dl_features[idx], dtype=torch.float64)
        fat_dl_feature = torch.unsqueeze(fat_dl_feature, dim=0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        # return clinical_feature, tumor_dl_feature, fat_dl_feature, label
        return tumor_dl_feature, fat_dl_feature, label

