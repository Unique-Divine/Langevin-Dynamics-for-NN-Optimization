import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from torch.nn.modules import loss; 
import torch.utils.data 
from torch.utils.data import dataset 
import os, sys
from torch import Tensor

class LitFFNN(pl.LightningModule):
    def __init__(self, loss_fn, optimizing_fn, 
                 mode='classifier', num_hidden_layers: int = 1, hidden_dim: int = None, 
                 architecture_shape: str = 'block', 
                 input_dim: int = None,
                 example_input: Tensor = None, num_classes: int = None):
        super().__init__()
        self.loss_function = loss_fn
        self.optimizing_fn = optimizing_fn
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.example_input = example_input
        self.num_classes = num_classes

        if mode in ['classifier', 'c']:
            assert num_classes is not None
            assert isinstance(num_classes, int)
            self.mode = 'classifier'

        # Layer definitions
        self.layers = nn.ModuleList()
        def RegularizedLinear(in_dim, out_dim) -> nn.Sequential:  
            return nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.ReLU(),
                nn.Dropout(p=0.15))

        def set_input_layer():
            input_layer = RegularizedLinear(
                in_dim=self.input_dim, out_dim=self.hidden_dim)
            self.layers.append(input_layer)
        
        def set_hidden_layers():
            for layer_idx in range(1, num_hidden_layers+1):
                hidden_layer = RegularizedLinear(
                    in_dim=self.hidden_dim, out_dim=self.hidden_dim)
                self.layers.append(hidden_layer)

        def set_output_layer():
            output_layer = nn.Linear(in_features=self.hidden_dim, 
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
        optimizer = self.optimizing_fn(params=self.parameters())
        return optimizer

    @property
    def input_dim(self) -> int:

        def init_input_dim():
            if self.example_input is not None:
                self._input_dim = self.example_input.flatten().shape[0]
            else: 
                raise NotImplementedError("example_input is None")
        
        def valid_input_dim() -> bool:
            input_dim_exists: bool = self._input_dim is not None 
            if input_dim_exists:
                input_dim_is_valid: bool = (
                    input_dim_exists and isinstance(self._input_dim, int))
            else:
                input_dim_is_valid: bool = False
            return input_dim_is_valid

        try: 
            assert valid_input_dim()
            return self._input_dim
        except:
            init_input_dim()
            return self._input_dim

    @property
    def hidden_dim(self) -> int:
        def init_hidden_dim():
            input_dim = self.input_dim 
            if self.mode == 'classifier':
                self._hidden_dim = round(
                    np.sqrt(input_dim * self.num_classes))
            elif self.mode == 'regressor':
                # self._hidden_dim
                raise NotImplementedError 
        try: 
            hidden_dim_exists: bool = self._hidden_dim is not None
            assert hidden_dim_exists
            assert isinstance(self._hidden_dim, int)
            return self._hidden_dim
        except:
            init_hidden_dim()
            return self._hidden_dim


    # --------------- Training and validation steps --------------- #
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'classifier':
            logits = self(x)
            loss = self.loss_function(logits, y)
        elif self.mode == 'regressor':
            raise NotImplementedError
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'classifier':
            logits = self(x)
            loss = self.loss_function(logits, y)
        elif self.mode == 'regressor':
            raise NotImplementedError
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)