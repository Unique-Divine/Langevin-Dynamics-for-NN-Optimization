# untested - IGNORE this module
import torch.nn as nn

class FFNN(nn.Module):
    """ Currently, the feed forward neural network is a part of the lightning 
    module. I'd like to separate the neural network out from thr training 
    procedure at some point, so I began moving some of the logic from 
    LitFFNN (class) in lit_modules.py to this file. """
    def __init__(self):
        self.num_hidden_layers: int # TODO: not declared
        self.num_classes: int # TODO: not declared

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