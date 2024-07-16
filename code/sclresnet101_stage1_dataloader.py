from PIL import Image
from torch.utils import data


class SIMCLRDataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        img = Image.open(self.data_path[index]).convert("RGB")
        i = self.data_path[index][13:]
        i = i[:-4]
        if self.transform is not None:
            imgL = self.transform(img)
            imgR = self.transform(img)

        return imgL, imgR



