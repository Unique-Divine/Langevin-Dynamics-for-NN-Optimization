# data modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
warnings.filterwarnings('ignore')
import torch.utils.data 
from torch.utils.data import dataset 
import torchvision
import os
from torch import Tensor

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = os.path.join(os.getcwd(), 'data'),
                 batch_size: int = 100):
        super().__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size

    def prepare_data(self):
        data_dir = self.data_dir
        torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        data_dir = self.data_dir
        transform = torchvision.transforms.ToTensor()

        if stage in ['fit', None]:
            train_data = torchvision.datasets.MNIST(
                root=data_dir, train=True, transform=transform)
            self.train_data, self.val_data = dataset.random_split(
                dataset=train_data, lengths = [54000, 6000])

        if stage in ['test', None]:
            test_data = torchvision.datasets.MNIST(
                root=data_dir, train=False, transform=transform)
            self.test_data = test_data 
        
    def get_dataloader(self, set: str = None):
        if set == "train":
            dl = torch.utils.data.DataLoader(
                self.train_data, batch_size=self.batch_size)
        elif set == "val":
            dl = torch.utils.data.DataLoader(
                self.val_data, batch_size=self.batch_size)
        elif set == "test":
            dl = torch.utils.data.DataLoader(
                self.test_data, batch_size=self.batch_size)
        else:
            raise ValueError() # TODO: Write error message.
        return dl

    def train_dataloader(self):
        return self.get_dataloader(set='train')

    def val_dataloader(self):
        return self.get_dataloader(set='val')
        
    def test_dataloader(self):
        return self.get_dataloader(set='test')
