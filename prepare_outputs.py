import matplotlib.pyplot as plt
import numpy as np

import hyper_parameters as hp

#   Function to show loss change during training
def loss_oscilation(losses, opt, epochs, lr, split_rate=hp.train_size_rate):
    plt.figure(figsize=(15, 7))

    plot_rage = int(len(losses)//2)

    x_axis = list(range(plot_rage, len(losses)))
    plt.scatter(x_axis, losses[plot_rage:], c="g", s=4, label="Testing data")

    coeffs = np.polyfit(x_axis, losses[plot_rage:], 1)
    trendline = np.polyval(coeffs, x_axis)
    plt.plot(x_axis, trendline, color='r', label='Trendline')

    plt.legend(prop={"size": 14})
    plt.grid()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Oscillation Trends\nOptimizer: {opt},Epoch: {epochs}\nLR: {lr}\nTraining rate: {split_rate}')
    plt.show()
    
    #file_name = f"{lr}_{epochs}_{opt}_{hp.today}"
    #plt.savefig(f"plots/{hp.MODEL_NAME}_loss_trends_{file_name}.jpg")

#   Function to show result comparison in plot
def plot_predictions(target_data, loss, opt, epochs, lr, split_rate = hp.train_size_rate, predictions=None):
    plt.figure(figsize=(15, 7))
    loss = round(loss, 2)
    info_text = f"Epoch: {epochs}\nLR: {lr}\nTraining rate: {split_rate}\nLoss: {loss}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

    x_axis = list(range(len(target_data)))
    target_data = target_data.detach().numpy()
    plt.scatter(x_axis, target_data, c="g", s=4, label="Ground Truth")
    
    if predictions is not None:
        predictions = predictions.detach().numpy()
        plt.scatter(x_axis, predictions, c="r", s=4, label="Predictions")

    file_name = f"{lr}_{epochs}_{opt}_{hp.today}"
    plt.legend(prop={"size": 14})
    plt.xlabel('Prediction number')
    plt.ylabel('Prediction values')
    plt.title(f'Prediction comparison to target values')
    plt.grid()
    #plt.savefig(f"plots/{hp.MODEL_NAME}_predictions_{file_name}.jpg")
    plt.show()

def scatter_plot(target_data, loss, opt, epochs, lr, split_rate = hp.train_size_rate, predictions=None):
    plt.figure(figsize=(15, 7))
    loss = round(loss, 2)
    info_text = f"Epoch: {epochs}\nLR: {lr}\nTraining rate: {split_rate}\nLoss: {loss}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

    target_data = target_data.detach().numpy()
    plt.scatter(predictions, target_data, c="g", s=4, label="Ground Truth")

    file_name = f"{lr}_{epochs}_{opt}_{hp.today}"
    plt.legend(prop={"size": 14})
    plt.xlabel('Prediction')
    plt.ylabel('Target data')
    plt.title(f'Scatter plot')
    plt.grid()
    #plt.savefig(f"plots/{hp.MODEL_NAME}_predictions_{file_name}.jpg")
    plt.show()

def create_graph_of_material_change_over_time(time, material, time_ref=None, material_ref=None):
    fig, ax = plt.subplots()

    if time_ref is not None and material_ref is not None:
        ax.scatter(time_ref, material_ref, color='red', marker='s', s=20, label='Sampled Data -> the twentieth sampe')
    ax.plot(time, material, linestyle='-', color='blue')

    sampled_time = time[::20]
    sampled_material = material[::20]

    ax.scatter(sampled_time, sampled_material, color='green', marker='s', s=20, label='Sampled Data -> the twentieth sampe')

    for i, txt in enumerate(sampled_material):  
        if isinstance(txt, np.ndarray):
            truncated_value = round(txt[0], 2)
        else:
            truncated_value = round(txt, 2)
        ax.annotate(truncated_value, (sampled_time[i], txt), textcoords="offset points", xytext=(0, 10), ha='center')
    
    
    ax.set_title('Oxidation Curve')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Mass Change (mg/cm²)')
    ax.grid(True)
    ax.legend()
    plt.show()