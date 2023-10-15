import matplotlib.pyplot as plt
import numpy as np

import hyper_parameters as hp

#   Function to show loss change during training
def loss_oscilation(losses, opt="Adam", epochs=hp.num_epochs, lr=hp.lr):
    plt.figure(figsize=(15, 7))
    info_text = f"File: {hp.MODEL_NAME}\nEpoch: {hp.num_epochs}\nLR: {lr}\nTraining rate: {hp.train_size_rate}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

    x_axis = list(range(int(hp.num_epochs/2), len(losses)))
    plt.scatter(x_axis, losses[int(hp.num_epochs/2):], c="g", s=4, label="Testing data")

    coeffs = np.polyfit(x_axis, losses[int(hp.num_epochs/2):], 1)
    trendline = np.polyval(coeffs, x_axis)
    plt.plot(x_axis, trendline, color='r', label='Trendline')

    plt.legend(prop={"size": 14})
    plt.grid()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Oscillation Trends\nOptimizer: {opt}')
    
    file_name = f"{lr}_{epochs}_{opt}_{hp.today}"
    plt.savefig(f"plots/{hp.MODEL_NAME}_loss_trends_{file_name}.jpg")

#   Function to show result comparison in plot
def plot_predictions(test_data, predictions=None, opt="Adam", epochs=hp.num_epochs, lr=hp.lr):
    plt.figure(figsize=(15, 7))
    info_text = f"File: {hp.MODEL_NAME}\nEpoch: {hp.num_epochs}\nLR: {lr}\nTraining rate: {hp.train_size_rate}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

    x_axis = list(range(len(test_data)))
    plt.scatter(x_axis, test_data, c="g", s=4, label="Ground Truth")

    if predictions is not None:
        plt.scatter(x_axis, predictions, c="r", s=4, label="Predictions")

    file_name = f"{lr}_{epochs}_{opt}_{hp.today}"
    plt.legend(prop={"size": 14})
    plt.grid()
    plt.savefig(f"plots/{hp.MODEL_NAME}_predictions_{file_name}.jpg")
    plt.show()

def create_graph_of_material_change_over_time(time, material):
    plt.plot(time, material, linestyle='-', color='red', marker='', label='Trend')

    plt.plot(time, material, linestyle='-', color='red', marker='', label='Trend')

    sampled_time = time[::20]
    sampled_material = material[::20]
    plt.scatter(sampled_time, sampled_material, color='blue', marker='s', s=50)  # 's' oznacza kwadrat
    plt.title(f'Oxidation curve')
    plt.xlabel('Time (h)')
    plt.ylabel('Mass change (mg/cm^2)')
    plt.grid(True)
    plt.legend()

    for i, txt in enumerate(material):
        if i % 20 == 0:
            value = material[i]
            formatted_value = f'{value}'
            plt.annotate(formatted_value, (time[i], value), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.show()