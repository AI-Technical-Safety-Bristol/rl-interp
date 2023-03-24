import torch
from torch import nn
import torch.nn.functional as F
from goofy import MultiLayerFeedForward

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class ProbedFeedForward(nn.Module):
    def __init__(self, n_layers, n_inputs, d_model, n_outputs):
        super().__init__()
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

def show_act(inp, probe, out, fig, ax):
    """
    Show the activations stored in probe, with output in out, on the plot plt
    The activations should be plotted as a 2D black and white image
    """
    fig.suptitle("activations")

    ax[0].imshow(inp.unsqueeze(0).detach().numpy(), cmap="gray")
    ax[0].set_xlabel("neuron")
    ax[0].set_yticks([])

    ax[1].imshow(probe.flip(0).detach().numpy(), cmap="gray")
    ax[1].set_xlabel("neuron")
    ax[1].set_ylabel("layer")
    ax[1].set_yticks([])

    ax[2].imshow(out.unsqueeze(0).detach().numpy(), cmap="gray")
    ax[2].set_xlabel("neuron")
    ax[2].set_yticks([])

def show_weights(net, fig, ax):
    """
    Show the weights of the network net
    The weights shoud be presented as a 2D black and white image per layer
    All images should be shown on the same figure
    Each image should have axis titles indicating to and from neurons
    """
    fig.suptitle("weights")

    ax[0].imshow(net.in_layer.weight.detach().numpy(), cmap="gray")
    ax[0].set_xlabel("from")
    ax[0].set_ylabel("to")
    ax[0].set_title("input proj")

    for i, l in enumerate(net.layers):
        ax[i+1].imshow(l.weight.detach().numpy(), cmap="gray")
        ax[i+1].set_xlabel("from")
        ax[i+1].set_ylabel("to")
        ax[i+1].set_title(f"layer {i} proj")

    ax[-1].imshow(net.out_layer.weight.transpose(0, 1).detach().numpy(), cmap="gray")
    ax[-1].set_xlabel("to")
    ax[-1].set_ylabel("from")
    ax[-1].set_title("output proj")

model = MultiLayerFeedForward(2, 8, 8, 4)
model.load_state_dict(torch.load("policy_100K.pt"))
model = ProbedFeedForward.from_mlff(model)

inp = torch.zeros(8)
out = model(inp)

weight_fig, weight_ax = plt.subplots(1, len(model.layers)+2)

show_weights(model, weight_fig, weight_ax)

act_fig = plt.figure()
gs = GridSpec(6, 2)

ac0 = act_fig.add_subplot(gs[0, 1])
ac1 = act_fig.add_subplot(gs[1:5, 1])
ac2 = act_fig.add_subplot(gs[5, 1])

act_ax = [ac0, ac1, ac2]

show_act(inp, model.probe, out, act_fig, act_ax)

slider_ax = [act_fig.add_subplot(gs[i, 0]) for i in range(6)]

# 8 sliders to set the input values
# high values: [1.5, 1.5, 5., 5., 3.14, 5., 1., 1. ]
# low values: [-1.5, -1.5, -5., -5., -3.14, -5., -0., -0. ]

x_coord = Slider(slider_ax[0], 'x', -1.5, 1.5, valinit=0.0)
y_coord = Slider(slider_ax[1], 'y', -1.5, 1.5, valinit=0.0)
x_vel = Slider(slider_ax[2], 'x_vel', -5., 5., valinit=0.0)
y_vel = Slider(slider_ax[3], 'y_vel', -5., 5., valinit=0.0)
angle = Slider(slider_ax[4], 'angle', -3.14, 3.14, valinit=0.0)
angle_vel = Slider(slider_ax[5], 'angle_vel', -5., 5., valinit=0.0)
#left_leg = Slider(act_ax[-2], 'left_leg', -1., 1., valinit=0.0)
#right_leg = Slider(act_ax[-1], 'right_leg', -1., 1., valinit=0.0)

def update(val, fig, ax):
    inp = torch.tensor([x_coord.val, y_coord.val, x_vel.val, y_vel.val, angle.val, angle_vel.val, 0, 0], dtype=torch.float32)
    out = model(inp)
    show_act(inp, model.probe, out, fig, ax)

x_coord.on_changed(lambda val: update(val, act_fig, act_ax))
y_coord.on_changed(lambda val: update(val, act_fig, act_ax))
x_vel.on_changed(lambda val: update(val, act_fig, act_ax))
y_vel.on_changed(lambda val: update(val, act_fig, act_ax))
angle.on_changed(lambda val: update(val, act_fig, act_ax))
angle_vel.on_changed(lambda val: update(val, act_fig, act_ax))

plt.show()