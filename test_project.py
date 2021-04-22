import torch 
import torch.nn as nn
import pytorch_lightning as pl
import data_modules
import lit_modules
import optimization
import warnings
warnings.filterwarnings('ignore')

class TestMNISTOptimizers:
    def get_test_lr(self) -> float:
        return 1e-3

    def optimizing_fns(self) -> dict: 
        optimizing_fns = dict(
            Adam = lambda params: torch.optim.Adam(
                params=params, lr=self.get_test_lr()),
            SGD = lambda params: torch.optim.SGD(
                params=params, lr=self.get_test_lr()),
            SGLD = lambda params: optimization.SGLD(
                params=params, lr=self.get_test_lr()),
            pSGLD = lambda params: optimization.PreconditionedSGLD(
                params=params, lr=self.get_test_lr()),
            )
        return optimizing_fns

    def test_quick_pass(self, optimizing_fn = None):
        optimizing_fn = optimizing_fn 
        if optimizing_fn is None:
            optimizing_fn = self.optimizing_fns()['Adam']

        data_module = data_modules.MNISTDataModule()
        mnist_img_dims = (1, 28, 28)
        channels, width, height = mnist_img_dims        
        network = lit_modules.LitFFNN(
            loss_fn=nn.CrossEntropyLoss(), 
            optimizing_fn=optimizing_fn, 
            mode='classifier', 
            num_hidden_layers=1, 
            num_classes=10, 
            input_dim = channels * width * height
        )
        trainer = pl.Trainer(gpus=0, fast_dev_run=True)
        trainer.fit(network, datamodule=data_module)
    
    def test_SGD(self):
        self.test_quick_pass(optimizing_fn=self.optimizing_fns()['SGD'])

    def test_SGLD(self):
        self.test_quick_pass(optimizing_fn=self.optimizing_fns()['SGLD'])
                        
    def test_pSGLD(self):
        self.test_quick_pass(optimizing_fn=self.optimizing_fns()['pSGLD'])

"""
Hide pytest warnings: https://stackoverflow.com/a/50821160/13305627

pytest -p no:warnings
"""