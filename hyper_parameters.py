import sys
from datetime import date

#   Hyper parameters
train_size_rate = 0.2           #   Default train size rate
lr = 0.001                       #   Default Learning Rate
num_epochs = 1500               #   Default Number Of Epochs
optimizer_arg = "default"       #   Default optimize / default = Adam
batch_size = 200000             #   Default batch size

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
DATA_FILE = "data.xlsx"
SEED = 32
TEST_DATA = "test_data.xlsx"

#   Temporarly redundant parameters
cross_validation_num = 5
patience = 0.1 * num_epochs
data_type = "float32"