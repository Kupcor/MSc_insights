import matplotlib.pyplot as plt

from torchviz import make_dot

import hyper_parameters as hp
import torch

def visualize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param_cpu = param.data.cpu().numpy()
            plt.figure(figsize=(10, 5))
            plt.title(f'Weights Distribution for {name}')
            plt.hist(param_cpu.flatten(), bins=50)
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

def plot_feature_importance(model):
    visualize_weights(model)