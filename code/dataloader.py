import torch
from torch.utils import data
from PIL import Image
import os
import pandas


class Dataset(data.Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        img = Image.open(self.data_path[index]).convert("RGB")
        i = self.data_path[index][13:]
        i = i[:-4]
        label_data = pandas.read_csv(os.path.join(self.label_path), encoding='gb18030')
        label = torch.tensor(label_data.loc[label_data['病历号'] == int(i), ['group1']].values.item())
        if self.transform is not None:
            img = self.transform(img)

        return img, label
