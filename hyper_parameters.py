import sys
from datetime import date
import torch

#   Hyper parameters for model training
train_size_rate = 0.2           #   Default train size rate | Used in training data splitting
lr = 0.001                      #   Default Learning Rate
num_epochs = 1500               #   Default Number Of Epochs
optimizer_arg = "default"       #   Default optimize / default = Adam
batch_size = 200000             #   Default batch size
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

#   Additional training Meta Data
SEED = 32
USE_CUDA = torch.cuda.is_available()
#   Use CUDA / GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Set: {gpu_name}")
else:
    device = torch.device("cpu")

#   Neuron numbers
input_layer_neurons = 4096*2
first_layer_neurons = 9192*2
second_layer_neurons = 2048*2
third_layer_neurons = 1024*2
fourth_layer_neurons = 512
output_layer_neurons = 256

#   Meta data
today = date.today()      
MODEL_NAME = sys.argv[0][:-3]
DATA_FILE = "data/data.xlsx"
TEST_DATA = "data/test_data.xlsx"
TENSOR_BOARD_DIR = f"tensor_board_logs/{MODEL_NAME}"


#   Temporarly redundant parameters
cross_validation_num = 5
patience = 0.1 * num_epochs
data_type = "float32"