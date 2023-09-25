import time
import sys
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ls

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

import sub_func.data_preparation as sfdp
use_cuda = torch.cuda.is_available()

#   23.08.2023 model_v9
#   We will try Transfer learning from previous models maybe


#   Notes

#   Results

#   Maybe TODO:
#   # Hyperparameter tuning
#   # Model architecture - testing
#   # TensorBoard - interesting
#   # Bayes optimalization

#   ____________________________    Pre Operations  _________________________________
#   Pre Data
MODEL_NAME = sys.argv[0][:-3]
today = date.today()        

#   HYPERPARAMETER
train_size_rate = 0.2           #   Default train size rate
lr = 0.01                       #   Default Learning Rate
num_epochs = 1000               #   Default Number Of Epochs
optimizer_arg = "default"       #   Default optimize / default = Adam
batch_size = 200000               #   Default batch size

#   Temporarly redundant parameters
cross_validation_num = 5
patience = 0.1 * num_epochs
data_type = "float32"

#   Tensor Board Directory
log_dir = f"runs/{MODEL_NAME}"

#   Just to have an option to set parameter via console ~ it is much simpler
#   LR
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

#   ____________________________    Select Optimizer  _________________________________
def select_optimizer(model, opt_arg=optimizer_arg, lr=lr):
    if opt_arg == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif opt_arg == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr)
    elif opt_arg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr)
    elif opt_arg == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    elif opt_arg == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr)
    elif opt_arg == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr)
    return optimizer

#   ____________________________    Load Data  _________________________________
#   Load training data
X_tensor, y_tensor = sfdp.get_prepared_data("data.xlsx")

#   Standard split (train and test data) - it is used in base training (without cross validation)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=train_size_rate,
                                                    random_state=42) #  Chce powtarzalności na danych na okres testowania

#   Input size
input_size = X_train.shape[1]

#   Create data sets
#kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, **kwargs)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)#, **kwargs)

#   ____________________________    Prediction Model  _________________________________
#   Main Architecture 
#   Maybe it is good idea to split it into another file
#   For now it has 3 hidden layer. I do not know yet if it is too small or to much, but for now works fine
#   I am also using batch normalization | Works fine, better than my implementation in data_preparation.py
#   For now using ReLU - probably I won't change it
class PredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, input_layer_neurons)
        self.bn_input = nn.BatchNorm1d(input_layer_neurons)
        self.first_hidden_layer = nn.Linear(input_layer_neurons, first_layer_neurons)
        self.bn_first = nn.BatchNorm1d(first_layer_neurons)
        self.second_hidden_layer = nn.Linear(first_layer_neurons, second_layer_neurons)
        self.bn_second = nn.BatchNorm1d(second_layer_neurons)
        self.third_hidden_layer = nn.Linear(second_layer_neurons, third_layer_neurons)
        self.bn_third = nn.BatchNorm1d(third_layer_neurons)
        self.fourth_hidden_layer = nn.Linear(third_layer_neurons, fourth_layer_neurons)
        self.bn_fourth = nn.BatchNorm1d(fourth_layer_neurons)
        self.fifth_hidden_layer = nn.Linear(fourth_layer_neurons, output_layer_neurons)
        self.output_layer = nn.Linear(output_layer_neurons, 1)
        self.relu = nn.ReLU()
        self.l_relu = nn.LeakyReLU()

    def forward(self, x):
        result = self.input_layer(x)
        result = self.bn_input(result)
        result = self.relu(result)
        result = self.first_hidden_layer(result)
        result = self.bn_first(result)
        result = self.relu(result)
        #result = self.l_relu(result)
        result = self.second_hidden_layer(result)
        result = self.bn_second(result)
        result = self.relu(result)
        result = self.third_hidden_layer(result)
        result = self.bn_third(result)
        result = self.relu(result)
        result = self.fourth_hidden_layer(result)
        result = self.bn_fourth(result)
        result = self.relu(result)
        result = self.fifth_hidden_layer(result)
        result = self.relu(result)
        result = self.output_layer(result)
        return result

#   ____________________________    Training Function  _________________________________
#   Main trainig function
def train_model(train_loader, model, loss_fun, opt_func, epochs):
    train_model_file_name = f"_p_{lr}_{num_epochs}_{opt_func.__class__.__name__}_{today}"
    writer = SummaryWriter(f"{log_dir}{train_model_file_name}")
    
    model.train()
    model.to(device)

    scheduler = ls.StepLR(opt_func, step_size=num_epochs/2, gamma=0.9)   
    loss_array = []

    with open(f'test_data/{MODEL_NAME}_{train_model_file_name}_loss_data.txt', "w") as file:
        start_time = time.time()

        for epoch in range(epochs + 1):
            epoch_loss_accumulator = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)
                opt_func.zero_grad()
                
                loss = loss_fun(predictions, batch_y.unsqueeze(1))
                loss.backward()
                opt_func.step()

                epoch_loss_accumulator += loss.item()
            
            average_epoch_loss = epoch_loss_accumulator / len(train_loader)
            
            if epoch % 100 == 0 or epoch == epochs:
                print(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {average_epoch_loss} | LR: {opt_func.param_groups[0]["lr"]} | Optimizer: {opt_func.__class__.__name__} | Device: {next(model.parameters()).device.type}')
                file.write(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()}\n')
                #print(epoch)
        
            loss_array.append(average_epoch_loss)
            scheduler.step()
                
            with torch.no_grad():
                test_pred, real_results, loss_value = test_model(test_loader, model, loss_fun, show_results=False)
                model.train()
                
            """ Temporarly turn off early stopping
            if epoch == 0:
                best_val_loss = loss_value
                epochs_since_improvement = 0

            if loss_value < best_val_loss:
                best_val_loss = loss_value
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # If validation loss hasn't improved for a certain number of epochs, stop training
            if epochs_since_improvement >= patience:
                print(f'Early stopping: No improvement in {patience} epochs.')
                break
            """
            #   Write to logs | Tensor Board
            writer.add_scalar("Loss/Train", average_epoch_loss, epoch)
            writer.add_scalar("Loss/Validation", loss_value, epoch)

        end_time = time.time()
        train_time = end_time - start_time
        file.write(f'Train time: {train_time}\n')
        writer.close()

    with open(f'test_data/{MODEL_NAME}_{train_model_file_name}_test_data.txt', "w") as file:
        file.write(f'Results in last iteration:\n')
        for i in range(len(predictions)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), batch_y[i].item())
            file.write(formatted_line)

    print(f'Training finished. Train time: {train_time}')
    return loss_array
    
#   Main test function
def test_model(test_loader, model,loss_fun,show_results=True, opt="Adam"):
    test_model_file_name = f"_p_{lr}_{num_epochs}_{opt.__class__.__name__}_{today}"

    model.eval()
    model.to(device)
    test_loss = 0.0
    all_predictions = []
    all_reall_results = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            batch_loss = loss_fun(predictions, batch_y.unsqueeze(1))
            test_loss += batch_loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_reall_results.append(batch_y.cpu().numpy())  
    
    average_test_loss = test_loss / len(test_loader)

    if show_results:
        print(f'Loss during test: {average_test_loss}')
        print(f'Test finished')
        with open(f'test_data/{MODEL_NAME}{test_model_file_name}_test_results_data.txt', "w") as file:
            file.write(f'Results in last iteration:\n')
            for i in range(len(predictions)):
                formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), batch_y[i].item())
                file.write(formatted_line)

    all_reall_results_flat = [item for sublist in all_reall_results for item in sublist]
    return all_predictions, all_reall_results_flat, average_test_loss

#   Function to show result comparison in plot
def plot_predictions(test_data=y_test, predictions=None, opt="Adam", epochs=num_epochs, lr=lr):
    plt.figure(figsize=(15, 7))
    info_text = f"File: {MODEL_NAME}\nEpoch: {num_epochs}\nLR: {lr}\nTraining rate: {train_size_rate}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)
    x_axis = list(range(len(test_data)))
    plt.scatter(x_axis, test_data, c="g", s=4, label="Ground Truth")

    if predictions is not None:
        plt.scatter(x_axis, predictions, c="r", s=4, label="Predictions")

    file_name_p2 = f"_p_{lr}_{epochs}_{opt}_{today}"
    plt.legend(prop={"size": 14})
    plt.grid()
    #plt.show()
    plt.savefig(f"pngs/losses{file_name_p2}.jpg")


#   Function to show loss change during training
def loss_oscilation(losses, opt="Adam", epochs=num_epochs, lr=lr):
    plt.figure(figsize=(15, 7))
    info_text = f"File: {MODEL_NAME}\nEpoch: {num_epochs}\nLR: {lr}\nTraining rate: {train_size_rate}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

    x_axis = list(range(int(num_epochs/2), len(losses)))
    plt.scatter(x_axis, losses[int(num_epochs/2):], c="g", s=4, label="Testing data")

    coeffs = np.polyfit(x_axis, losses[int(num_epochs/2):], 1)
    trendline = np.polyval(coeffs, x_axis)
    plt.plot(x_axis, trendline, color='r', label='Trendline')

    plt.legend(prop={"size": 14})
    plt.grid()
    
    file_name_p2 = f"_p_{lr}_{epochs}_{opt}_{today}"
    plt.show()

    plt.savefig(f"pngs/loss_trend{file_name_p2}.jpg")


"""
Redundanty for now
#   Cross validation, similar to test model function, but it use cross validation
#   For now I do not want it to be saved in file
def cross_validation(X=X_tensor, y=y_tensor, loss_fn=nn.MSELoss(), num_epochs=num_epochs, num_folds=cross_validation_num):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_loss = []
    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold_num+1}/{num_folds}")
        
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]
        
        input_size = X_train.shape[1]
        Prediction_Model_CV = PredictionModel(input_size)  #  New instance of model for each fold

        optimizer = select_optimizer(Prediction_Model_CV)

        train_model(X_train_cv, y_train_cv, Prediction_Model_CV, loss_fn, optimizer, num_epochs)
        predictions, test_loss = test_model(X_val_cv, y_val_cv, Prediction_Model_CV, loss_fn)
        fold_loss.append(test_loss)
    print(fold_loss)
"""

#   Standard run
def run_model(train_loader = train_loader, test_loader = test_loader, num_epochs=num_epochs, lr=lr, opt_func=optimizer_arg):
    Prediction_model_v = PredictionModel(input_size)
    Prediction_model_v = Prediction_model_v.to(device)

    loss_fn = nn.MSELoss() 
    
    #   Select optimizer
    optimizer = select_optimizer(Prediction_model_v, opt_arg=opt_func, lr=lr)
    #   Traing model
    train_loss = train_model(train_loader, Prediction_model_v, loss_fn, optimizer, num_epochs)
    #   Test model
    predictions, real_results, test_loss = test_model(test_loader, Prediction_model_v, loss_fn, opt=opt_func)

    #   Show plots
    plot_predictions(test_data=real_results, predictions=predictions, opt=opt_func, lr=lr, epochs=num_epochs)
    loss_oscilation(train_loss, opt=opt_func, epochs=num_epochs, lr=lr)

    #   Save model
    file_name_p2 = f"_p_{lr}_{num_epochs}_{optimizer.__class__.__name__}_testLoss_{test_loss}_{today}"
    save_path = "models/" + MODEL_NAME + file_name_p2 + '.pth'
    torch.save(Prediction_model_v.state_dict(), save_path)

def run_loaded_model(file_path,X_val=X_test, Y_val=y_test, loss_fun = nn.MSELoss()):
    from tensorboardX import SummaryWriter
    dict = f"runs/{MODEL_NAME}_{file_path}"
    writer = SummaryWriter(dict)

    saved_model_path = file_path
    model = PredictionModel(input_size)
    model.load_state_dict(torch.load(saved_model_path))
    model = model.to(device)
    model.eval()
    predictions, reall_values, test_loss = test_model(test_loader=test_loader, model=model, loss_fun=nn.MSELoss(), opt=optimizer_arg, show_results=True)
    plot_predictions(test_data=reall_values, predictions=predictions)

    #sample_input = X_val[0].unsqueeze(0).clone().detach()
    #sample_input.to(device)
    #writer.add_graph(model, sample_input)
    writer.close()

run_model()
#cross_validation()
#run_loaded_model("models/model_v8_p_0.0001_30000_AdamW_testLoss_1.622692584991455_2023-08-28.pth")
