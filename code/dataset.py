import torch
import pandas as pd
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, data_path, train=True, test=False, transform=None):
        super(MnistDataset, self).__init__()
        self.data = pd.read_csv(data_path)
        self.test = test
        
        if train == True and not test:
            self.data = self.data[:-7000]
        
        elif test:
            self.data = self.data
        else:
            self.data = self.data[-7000:]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _read_data(self, idx):
        mnist_tensor = torch.zeros((784))
        for i in range(784):
            mnist_tensor[i] = self.data.iloc[idx]['pixel'+str(i)]
        mnist_tensor = torch.reshape(mnist_tensor, (1, 28, 28))
        sample = {
            "image" : mnist_tensor/255,
        }

        if self.test:
            pass
        else:
            label = self.data.iloc[idx]['label']
            sample['label'] = torch.tensor(label)
        
        if self.transform:
            sample["image"] = self.transform(sample["image"]).float()
            
        return sample

    def __getitem__(self, index):
        sample = self._read_data(index)
        return sample

