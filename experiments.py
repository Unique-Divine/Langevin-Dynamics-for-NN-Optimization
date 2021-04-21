import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
from torch.nn.modules import loss; warnings.filterwarnings('ignore')
import torch.utils.data 
from torch.utils.data import dataset 
import torchvision
import os
from torch import Tensor

class FFNN(pl.LightningModule):
    def __init__(self, loss_fn, optimizer, 
                 mode='classifier', num_hidden_layers: int = 1, hidden_dim: int = None, 
                 architecture_shape: str = 'block', 
                 example_input: Tensor = None, num_classes: int = None):
        super()._init__()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if mode in ['classifier', 'c']:
            assert num_classes is not None
            assert isinstance(num_classes, int)
            self.mode = 'classifier'

        # Layer definitions
        self.layers = []
        def RegularizedLinear(in_dim, out_dim) -> nn.Sequential:  
            return nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.ReLU(),
                nn.Dropout(p=0.15))

        def set_input_layer():
            if example_input is not None:
                input_dim = example_input.flatten().shape[0]
                input_layer = RegularizedLinear(
                    in_dim=input_dim, out_dim=hidden_dim)
            else:
                raise NotImplementedError("example_input is None")
            self.layers.append(input_layer)
        
        def set_hidden_layers():
            for layer_idx in range(1, num_hidden_layers+1):
                if hidden_dim is None and example_input is not None:
                    input_dim = example_input.flatten().shape[0]
                    if self.mode == 'classifier':
                        hidden_dim = round(np.sqrt(input_dim * num_classes))
                    elif self.mode == 'regressor':
                        hidden_dim: int = None
                        raise NotImplementedError 

                hidden_layer = RegularizedLinear(
                    in_dim=hidden_dim, out_dim=hidden_dim)
                self.layers.append(hidden_layer)

        def set_output_layer():
            output_layer = nn.Linear(in_features=hidden_dim, 
                                     out_features=num_classes)
            self.layers.append(output_layer)

        def set_layers():
            set_input_layer()
            set_hidden_layers()
            set_output_layer()

        set_layers()

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        if self.mode == 'classifier':
            logits = F.log_softmax(input=x, dim=1)
            return logits

    def configure_optimizers(self):
        return self.optimizer

    # --------------- Training and validation steps --------------- #
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'classifier':
            logits = self(x)
            loss = self.loss_fn(logits, y)
        elif self.mode == 'regressor':
            raise NotImplementedError
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'classifier':
            logits = self(x)
            loss = self.loss_fn(logits, y)
        elif self.mode == 'regressor':
            raise NotImplementedError
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = os.path.join(os.getcwd(), 'data'),
                 batch_size: int = 100):
        self.data_dir = data_dir 
        self.batch_size = batch_size

    def prepare_data(self):
        data_dir = self.data_dir
        torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    
    def setup(self):
        data_dir = self.data_dir
        
        mnist_train = torchvision.datasets.MNIST(
            root=data_dir, train=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(
            root=data_dir, train=False, transform=transform)
        
        self.mnist_train, self.mnist_val = dataset.random_split(
            dataset=mnist_train, lengths = [54000, 6000])
        self.mnist_test = mnist_test

    def train_dataloader(self):
        mnist_train_dl = torch.utils.data.DataLoader(
            self.mnist_train, batch_size=self.batch_size)
        return mnist_train_dl

    def val_dataloader(self):
        mnist_val_dl = torch.utils.data.DataLoader(
            self.mnist_val, batch_size=self.batch_size)
        return mnist_val_dl    
        
    def test_dataloader(self):
        mnist_test_dl = torch.utils.data.DataLoader(
            self.mnist_test, batch_size=self.batch_size)
        return mnist_test_dl

def example():
    data_module = MNISTDataModule()
    
    model = FFNN(loss_fn=, optimizer=, mode='classifier', num_hidden_layers=1, 
                 )
    
    trainer = Trainer(gpus=0)
    trainer.fit(model, data_module)


