import sys
from datetime import date
import torch

#   Hyper parameters for model training
train_size_rate = 0.2           #   Default train size rate | Used in training data splitting
lr = 0.001                      #   Default Learning Rate
num_epochs = 1500               #   Default Number Of Epochs
optimizer_arg = "default"       #   Default optimize / default = Adam
seed = 32                       #   Default seed
neurons_in_hidden_layers = [13] #   Default - one hidden layer with 13 neurons
x_scaler = None
y_scaler = None
activation_func = "ReLU"

#   Set hyper parameters if they will be provided
if len(sys.argv) > 1:
    print(f"Lr set to {sys.argv[1]}")
    lr = float(sys.argv[1]) 
#   Epoch numbers
if len(sys.argv) > 2:
    print(f"Epoch number set to {sys.argv[2]}")
    num_epochs = int(sys.argv[2])
#   Optimizer
if len(sys.argv) > 3:
    print(f"Optimizer set to {sys.argv[3]}")
    optimizer_arg = sys.argv[3]
if len(sys.argv) > 4:
    print(f"Seed set to {sys.argv[4]}")
    seed = int(sys.argv[4])
if len(sys.argv) > 5:
    print(f"Training size rate set to {sys.argv[5]}")
    train_size_rate = float(sys.argv[5])
    if (train_size_rate > 0.8 or train_size_rate < 0.1):
        train_size_rate = 0.3
if len(sys.argv) > 6:
    neurons_in_hidden_layers = []
    neurons = sys.argv[6].split(',')
    for neuron in neurons:
        neurons_in_hidden_layers.append(int(neuron))
    print(f"Hidden layers set to {neurons_in_hidden_layers} | Number of hidden layers: {len(neurons_in_hidden_layers)}")
if len(sys.argv) > 7:
    print(f"Activation func set to {sys.argv[7]}")
    activation_func = sys.argv[7]

#   Additional training Meta Data
SEED = seed
USE_CUDA = torch.cuda.is_available()
#   Use CUDA / GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Set: {gpu_name}")
else:
    device = torch.device("cpu")
    print(f'Set: cpu')

device = torch.device("cpu")

#   Meta data
today = date.today()      
MODEL_NAME = sys.argv[0][:-3]
DATA_FILE = "data/data_cleared.xlsx"
TESTING_FILE = "data/chart_data.xlsx"
TEST_DATA = "data/test_data.xlsx"
TENSOR_BOARD_DIR = f"tensor_board_logs/{MODEL_NAME}"
MODEL_FILE_NAME = f"model_{lr}_{num_epochs}_{optimizer_arg}_{today}"
FEATURES = "Temperature[C]", "Zr[at%]", "Nb[at%]", "Mo[at%]", "Cr[at%]", "Al[at%]", "Ti[at%]", "Ta[at%]", "W[at%]", "Time[h]"
PREDICTION = "Mass Change [kg.cm2]"

#   Temporarly redundant parameters
cross_validation_num = 5
patience = 0.1 * num_epochs
data_type = "float32"
batch_size = 64                 #   Default batch size

def set_hyperparameters_from_string(param_string):
    lr = 0.001  # Domyślna wartość współczynnika uczenia
    num_epochs = 10  # Domyślna liczba epok
    optimizer_arg = "adam"  # Domyślny optymalizator
    seed = 42  # Domyślny ziarno losowości
    train_size_rate = 0.3  # Domyślny stosunek rozmiaru treningowego
    neurons_in_hidden_layers = []  # Domyślne liczby neuronów w warstwach ukrytych
    activation_func = "ELU"

    params = param_string.split()

    if len(params) > 0:
        lr = float(params[0])
        print(f"Lr set to {lr}")

    if len(params) > 1:
        num_epochs = int(params[1])
        print(f"Epoch number set to {num_epochs}")

    if len(params) > 2:
        optimizer_arg = params[2]
        print(f"Optimizer set to {optimizer_arg}")

    if len(params) > 3:
        seed = int(params[3])
        print(f"Seed set to {seed}")

    if len(params) > 4:
        train_size_rate = float(params[4])
        if train_size_rate > 0.8 or train_size_rate < 0.1:
            train_size_rate = 0.3
        print(f"Training size rate set to {train_size_rate}")

    if len(params) > 5:
        neurons = params[5].split(',')
        for neuron in neurons:
            neurons_in_hidden_layers.append(int(neuron))
        print(f"Hidden layers set to {neurons_in_hidden_layers} | Number of hidden layers: {len(neurons_in_hidden_layers)}")

    if len(params) > 6:
        print(f"Activation func set to {params[6]}")
        activation_func = params[6]

    return lr, num_epochs, optimizer_arg, seed, train_size_rate, neurons_in_hidden_layers, activation_func