import torch
from torch import nn
import torch.nn.functional as F
from goofy import MultiLayerFeedForward

class ProbedFeedForward(nn.Module):
    def __init__(self, n_layers, n_inputs, d_model, n_outputs):
        super().__init__()
        self.hyperparams = [n_layers, n_inputs, d_model, n_outputs]
        self.in_layer = nn.Linear(n_inputs, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.out_layer = nn.Linear(d_model, n_outputs)
        
        self.register_buffer("probe", torch.zeros(n_layers+1, d_model))

    def forward(self, x):
        x = F.gelu(self.in_layer(x))
        self.probe[0,:] = x
        for i, l in enumerate(self.layers):
            x = F.gelu(l(x))
            self.probe[i+1,:] = x
        x = self.out_layer(x)
        return x

    def from_mlff(net):
        """
        Create a ProbedFeedForward from a MultiLayerFeedForward
        """
        pff = ProbedFeedForward(len(net.layers), net.in_layer.in_features, net.layers[0].out_features, net.out_layer.out_features)
        pff.in_layer.weight = net.in_layer.weight
        pff.in_layer.bias = net.in_layer.bias
        for i, l in enumerate(net.layers):
            pff.layers[i].weight = l.weight
            pff.layers[i].bias = l.bias
        pff.out_layer.weight = net.out_layer.weight
        pff.out_layer.bias = net.out_layer.bias
        return pff
    
    def gen_sparse_model(self, w_cutoff=0.1, b_cutoff=0.1):
        """
        Generate a sparse model from current weights and biases
        Set values below the cutoff to zero
        """
        sparse_model = ProbedFeedForward(*self.hyperparams)
        # generate a new state_dict for the sparse model
        sparse_state_dict = {}
        for k, v in self.state_dict().items():
            sparse_state_dict[k] = v.clone()
            # if k is a weight, set values below cutoff to zero
            if k.endswith("weight"):
                sparse_state_dict[k][v.abs() < w_cutoff] = 0
            # if k is a bias, set values below cutoff to zero
            elif k.endswith("bias"):
                sparse_state_dict[k][v.abs() < b_cutoff] = 0
        # load the sparse state dict into the sparse model
        sparse_model.load_state_dict(sparse_state_dict)
        return sparse_model
    
    def gen_sparse_model_proportion(self, w_percent=0.1, b_percent=0.1):
        """
        Generate a sparse model from current weights and biases
        Set values below the cutoff to zero
        """
        sparse_model = ProbedFeedForward(*self.hyperparams)
        # generate a new state_dict for the sparse model
        sparse_state_dict = {}
        for k, v in self.state_dict().items():
            sparse_state_dict[k] = v.clone()
            # if k is a weight, set values below cutoff to zero
            if k.endswith("weight"):
                try:
                    cutoff = torch.topk(v.abs().flatten(), int(v.numel()*w_percent))[0][-1]
                except:
                    cutoff = 0
                sparse_state_dict[k][v.abs() < cutoff] = 0
            # if k is a bias, set values below cutoff to zero
            elif k.endswith("bias"):
                try:
                    cutoff = torch.topk(v.abs().flatten(), int(v.numel()*b_percent))[0][-1]
                except:
                    cutoff = 0
                sparse_state_dict[k][v.abs() < cutoff] = 0
        # load the sparse state dict into the sparse model
        sparse_model.load_state_dict(sparse_state_dict)
        return sparse_model