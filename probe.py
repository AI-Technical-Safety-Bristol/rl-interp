import torch
from torch import nn
import torch.nn.functional as F
from goofy import MultiLayerFeedForward
import math

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.gridspec import GridSpec
from probed import ProbedFeedForward
import argparse

# parse optional arguments input model, model dimensions (num hidden layers, input dim, hidden dim, output dim), and sparsity levels for weights and biases
parser = argparse.ArgumentParser(description='Probe a model')
parser.add_argument('--model', type=str, default='policy.pt', help='path to model')
parser.add_argument('--test', type=bool, default=False, help='test with random model')
parser.add_argument('--dims', type=int, nargs='+', default=[2, 8, 8, 4], help='dimensions of model')
parser.add_argument('--w_sparsity', type=float, default=1, help='sparsity of weights')
parser.add_argument('--b_sparsity', type=float, default=1, help='sparsity of biases')
args = parser.parse_args()

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
    The weights should be presented as a 2D black and white image per layer
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

def show_biases(net, fig, ax):
    """
    Show the biases of the network net
    The biases should be presented as a 2D black and white image per layer
    All images should be shown on the same figure
    Each image should have axis titles indicating to and from neurons
    """
    fig.suptitle("weights")

    ax[0].imshow(net.in_layer.bias.unsqueeze(0).transpose(0,1).detach().numpy(), cmap="gray")
    ax[0].set_xlabel("from")
    ax[0].set_ylabel("to")
    ax[0].set_title("input proj")

    for i, l in enumerate(net.layers):
        ax[i+1].imshow(l.bias.unsqueeze(0).transpose(0,1).detach().numpy(), cmap="gray")
        ax[i+1].set_xlabel("from")
        ax[i+1].set_ylabel("to")
        ax[i+1].set_title(f"layer {i} proj")

    ax[-1].imshow(net.out_layer.bias.unsqueeze(0).transpose(0,1).detach().numpy(), cmap="gray")
    ax[-1].set_xlabel("to")
    ax[-1].set_ylabel("from")
    ax[-1].set_title("output proj")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not args.test:
    model = MultiLayerFeedForward(*args.dims)
    model.load_state_dict(torch.load(args.model))
    model = ProbedFeedForward.from_mlff(model).to(device)
    model = model.gen_sparse_model_proportion(args.w_sparsity, args.b_sparsity)
else:
    model = ProbedFeedForward(*args.dims).to(device)
    model = model.gen_sparse_model_proportion(args.w_sparsity, args.b_sparsity)

inp = torch.zeros(args.dims[1])
out = model(inp)

weight_fig, weight_ax = plt.subplots(1, len(model.layers)+2)

show_weights(model, weight_fig, weight_ax)

bias_fig, bias_ax = plt.subplots(1, len(model.layers)+2)

show_biases(model, bias_fig, bias_ax)

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

all_lines = []
def draw_line(fig, c0, c1, col):
    global all_lines
    # draw a line from x to y with color dependent on weights
    # append line to lines
    line = lines.Line2D([c0[0],c1[0]], [c0[1],c1[1]], lw=2, color=col)
    fig.add_artist(line)
    all_lines.append(line)

def ax_to_fig_coord(fig, ax, coord):
    # convert matplotlib ax coords to fig coords
    return fig.transFigure.inverted().transform(ax.transData.transform(coord))

def fig_to_ax_coord(fig, ax, coord):
    return ax.transData.inverted().transform(fig.transFigure.transform(coord))

# on mouseover, draw lines from hovered neuron to all other connected neurons
def on_hover(event):
    # nn.Linear(a,b) represented as a matrix of size (b,a)
    # weight from x->y given by l.weight[y,x]

    global all_lines
    # clear all lines
    for line in all_lines:
        line.remove()
    all_lines = []
    if event.inaxes == act_ax[1]:
        # get neuron coordinates
        x, y = event.xdata, event.ydata
        x, y = round(x), round(y)
        # get weights and coords of all neurons feeding into neuron x,y
        source = []
        if y == 0:
            # input layer
            source = [(ax_to_fig_coord(act_fig, act_ax[0], (z,0)), w) for z, w in enumerate(model.in_layer.weight[x,:])]
        else:
            # hidden layer
            source = [(ax_to_fig_coord(act_fig, act_ax[1], (z,y-1)), w) for z, w in enumerate(model.layers[y-1].weight[x,:])]
        # get weights and coords of all neurons that neuron x,y feeds into
        sink = []
        if y == len(model.layers):
            # output layer
            sink = [(ax_to_fig_coord(act_fig, act_ax[2], (z,0)), w) for z, w in enumerate(model.out_layer.weight[:,x])]
        else:
            # hidden layer
            sink = [(ax_to_fig_coord(act_fig, act_ax[1], (z,y+1)), w) for z, w in enumerate(model.layers[y].weight[:,x])]
        
        # draw lines
        all_w = [w.item() for _, w in source] + [w.item() for _, w in sink]
        max_w = abs(max(all_w)) + 0.01
        min_w = abs(min(all_w)) + 0.01

        for c0, w in source:
            w = w.item()
            w_ = w / min_w if w < 0 else w / max_w
            r,g,b,_ = plt.cm.RdBu(w_)
            a = 1 - math.exp(-w**2)
            draw_line(act_fig, c0, ax_to_fig_coord(act_fig, act_ax[1], (x, y)), (r,g,b,a))
        for c1, w in sink:
            w = w.item()
            w_ = w / min_w if w < 0 else w / max_w
            r,g,b,_ = plt.cm.RdBu(w_)
            a = 1 - math.exp(-w**2)
            draw_line(act_fig, ax_to_fig_coord(act_fig, act_ax[1], (x, y)), c1, (r,g,b,a))
        act_fig.canvas.draw_idle()
    elif event.inaxes == act_ax[0]:
        # get neuron coordinates
        x = event.xdata
        x = round(x)
        # only feed from
        sink = [(ax_to_fig_coord(act_fig, act_ax[1], (z,0)), w) for z, w in enumerate(model.in_layer.weight[:,x])]
        # draw lines
        all_w = [w.item() for _, w in sink]
        max_w = abs(max(all_w)) + 0.01
        min_w = abs(min(all_w)) + 0.01

        for c0, w in sink:
            w = w.item()
            w_ = w / min_w if w < 0 else w / max_w
            r,g,b,_ = plt.cm.RdBu(w_)
            a = 1 - math.exp(-w**2)
            draw_line(act_fig, c0, ax_to_fig_coord(act_fig, act_ax[0], (x,0)), (r,g,b,a))
        act_fig.canvas.draw_idle()
    elif event.inaxes == act_ax[2]:
        # get neuron coordinates
        x = event.xdata
        x = round(x)
        # only feed into
        source = [(ax_to_fig_coord(act_fig, act_ax[1], (z,len(model.layers))), w) for z, w in enumerate(model.out_layer.weight[x,:])]
        # draw lines
        all_w = [w.item() for _, w in source]
        max_w = abs(max(all_w)) + 0.01
        min_w = abs(min(all_w)) + 0.01

        for c1, w in source:
            w = w.item()
            w_ = w / min_w if w < 0 else w / max_w
            r,g,b,_ = plt.cm.RdBu(w_)
            a = 1 - math.exp(-w**2)
            draw_line(act_fig, ax_to_fig_coord(act_fig, act_ax[2], (x,0)), c1, (r,g,b,a))
        act_fig.canvas.draw_idle()

act_fig.canvas.mpl_connect("motion_notify_event", on_hover)

# delete lines on mouseout
def on_leave(event):
    global all_lines
    for line in all_lines:
        line.remove()
    act_fig.canvas.draw_idle()
    all_lines = []

act_fig.canvas.mpl_connect("axes_leave_event", on_leave)

plt.show()