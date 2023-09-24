import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from datetime import date

from torch.utils.tensorboard import SummaryWriter

import torch.optim.lr_scheduler as ls

import sub_func.data_preparation as sfdp

#   23.08.2023 model_v6
#   We will try Transfer learning from previous models
#   And add one more layer
#   Implemented TensorBoard
#   tensorboard --logdir=runs
#   TODO BIG: make few training with different hyper parameter and different optimizers

#   Notes

#   Results

#   Maybe TODO:
#   # Hyperparameter tuning
#   # Model architecture - testing
#   # TensorBoard - interesting
#   # Bayes optimalization

#   Hyperparameters and Pre-Data
MODEL_NAME = sys.argv[0][:-3]
train_size_rate = 0.2
lr = 0.01
num_epochs = 1000
cross_validation_num = 5
optimizer_arg = "default"
today = date.today()

#   Tensor Board
log_dir = f"runs/{MODEL_NAME}"

#   Load data to model
X_tensor, y_tensor = sfdp.get_prepared_data("data.xlsx")

#   Standard split (train and test data) - it is used in base training (without cross validation)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, 
                                                    y_tensor, 
                                                    test_size=train_size_rate,
                                                    random_state=42) #  Chce powtarzalności na danych na okres testowania

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
X_tensor = move_to_device(X_tensor, device)
y_tensor = move_to_device(y_tensor, device)

#   Just to have an option to set parameter via console ~ it is much simpler
if len(sys.argv) > 1:
    print(f"Lr set to {sys.argv[1]}")
    lr = float(sys.argv[1]) 

if len(sys.argv) > 2:
    print(f"Epoch number set to {sys.argv[2]}")
    num_epochs = int(sys.argv[2])

if len(sys.argv) > 3:
    print(f"Optimizer set to {sys.argv[3]}")
    optimizer_arg = sys.argv[3]

patience = 0.1 * num_epochs

#   Main Architecture 
#   Maybe it is good idea to split it into another file
#   For now it has 3 hidden layer. I do not know yet if it is too small or to much, but for now works fine
#   I am also using batch normalization | Works fine, better than my implementation in data_preparation.py
#   For now using ReLU - probably I won't change it
class PredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.bn_input = nn.BatchNorm1d(64)
        self.first_hidden_layer = nn.Linear(64, 512)
        self.bn_first = nn.BatchNorm1d(512)
        self.second_hidden_layer = nn.Linear(512, 256)
        self.bn_second = nn.BatchNorm1d(256)
        self.third_hidden_layer = nn.Linear(256, 100)
        self.bn_third = nn.BatchNorm1d(100)
        self.fourth_hidden_layer = nn.Linear(100, 30)
        self.output_layer = nn.Linear(30, 1)
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
        result = self.relu(result)
        result = self.output_layer(result)
        return result

#   Main trainig function
#   I wanted to retain some 'elasticity'
def train_model(X, y, model, loss_fun, opt_func, epochs):
    parameters = f"_p_{lr}_{num_epochs}_{opt_func.__class__.__name__}_{today}"
    writer = SummaryWriter(f"{log_dir}{parameters}")

    model.train()
    loss_array = []
    parameters = f"_p_{lr}_{num_epochs}_{opt_func.__class__.__name__}_{today}"
    with open(f'test_data/{MODEL_NAME}_{parameters}_loss_data.txt', "w") as file:
        start_time = time.time()
        #   Added learning schedulers
        #scheduler = ls.ExponentialLR(opt_func, gamma=1.6)
        scheduler = ls.StepLR(opt_func, step_size=num_epochs/2, gamma=0.9)

        for epoch in range(epochs + 1):
            predictions = model(X)
            opt_func.zero_grad()
            
            loss = loss_fun(predictions, y.unsqueeze(1))
            loss.backward()
            opt_func.step()

            loss_array.append(loss.item())

            if epoch % 100 == 0 or epoch == epochs:
                mape = mean_absolute_error(y.numpy(), predictions.detach().numpy()) * 100   #   Mean Abosulute Percentage Error
                rmse = np.sqrt(mean_squared_error(y.numpy(), predictions.detach().numpy())) #   Root Mean Squere Error
                print(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()} | LR: {opt_func.param_groups[0]["lr"]} | MAPE: {mape} | RMEA: {rmse} | Optimizer: {opt_func.__class__.__name__}')
                file.write(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()}\n')
            scheduler.step()
            
            
            #   Early stopping - first attemp
            with torch.no_grad():
                test_pred, loss_value = test_model(X_train, y_train, model, loss_fun, show_results=False)
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
                
            #   Write to logs
            writer.add_scalar("Loss/Train", loss.item(), epoch)
            writer.add_scalar("Loss/Validation", loss_value, epoch)

        end_time = time.time()
        train_time = end_time - start_time
        file.write(f'Train time: {train_time}\n')
        writer.close()

    parameters = f"_p_{lr}_{num_epochs}_{opt_func.__class__.__name__}_{today}"
    with open(f'test_data/{MODEL_NAME}_{parameters}_test_data.txt', "w") as file:
        file.write(f'Results in last iteration:\n')

        for i in range(len(predictions)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), y[i].item())
            file.write(formatted_line)

    print(f'Training finished. Train time: {train_time}')
    return loss_array
    
#   Main test function
def test_model(X,y,model,loss_fun,show_results=True, opt="Adam"):
    model.eval()
    predictions = model(X)
    loss = loss_fun(predictions, y.unsqueeze(1))

    parameters = f"_p_{lr}_{num_epochs}_{opt}_{today}"
    with open(f'test_data/{MODEL_NAME}{parameters}_test_results_data.txt', "w") as file:
        file.write(f'Results in last iteration:\n')

        for i in range(len(predictions)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), y[i].item())
            file.write(formatted_line)

    if show_results:
        print(f'Loss during test: {loss.item()}')
        print(f'Test finished')
    return predictions, loss.item()

#   Function to show result comparison in plot
def plot_predictions(test_data=y_test, predictions=None, opt="Adam", epochs=num_epochs, lr=lr):
  plt.figure(figsize=(15, 7))
  info_text = f"File: {MODEL_NAME}\nEpoch: {num_epochs}\nLR: {lr}\nTraining rate: {train_size_rate}"
  plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

  x_axis = list(range(len(test_data)))
  plt.scatter(x_axis, test_data, c="g", s=4, label="Testing data")

  if predictions is not None:
    plt.scatter(x_axis, predictions, c="r", s=4, label="Predictions")

  parameters = f"_p_{lr}_{epochs}_{opt}_{today}"
  plt.legend(prop={"size": 14})
  plt.grid()
  #plt.show()
  plt.savefig(f"pngs/losses{parameters}.jpg")


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
    parameters = f"_p_{lr}_{epochs}_{opt}_{today}"

    plt.legend(prop={"size": 14})
    plt.grid()
    
    
    plt.savefig(f"pngs/loss_trend{parameters}.jpg")
    #plt.show()

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

#   Standard run
#   I need to adjust it to "cross validation"
def run_model(X_tr=X_train, y_tr=y_train, X_ts = X_test, y_ts = y_test, num_epochs=num_epochs, lr=lr, opt_func="Adam"):
    input_size = X_tr.shape[1]
    Prediction_model_v = PredictionModel(input_size)

    #loss_fn = nn.L1Loss()  # L1
    loss_fn = nn.MSELoss()  # L2
    
    #   Select optimizer
    optimizer = select_optimizer(Prediction_model_v, opt_arg=opt_func, lr=lr)

    #   Traing model
    train_loss = train_model(X_tr, y_tr, Prediction_model_v, loss_fn, optimizer, num_epochs)

    #   Test model
    predictions, test_loss = test_model(X_ts, y_ts, Prediction_model_v, loss_fn, opt=opt_func)

    #   Show plots
    predictions = predictions.detach().numpy()
    plot_predictions(predictions=predictions, opt=opt_func, lr=lr, epochs=num_epochs)
    loss_oscilation(train_loss, opt=opt_func, epochs=num_epochs, lr=lr)

    #   Save model
    parameters = f"_p_{lr}_{num_epochs}_{optimizer.__class__.__name__}_testLoss_{test_loss}_{today}"
    save_path = "models/" + MODEL_NAME + parameters + '.pth'
    torch.save(Prediction_model_v.state_dict(), save_path)

def run_loaded_model(file_path,X_val=X_test, Y_val=y_test, loss_fun = nn.MSELoss()):
    from tensorboardX import SummaryWriter
    dict = f"runs/{MODEL_NAME}"
    writer = SummaryWriter(dict)

    saved_model_path = file_path
    input_size = X_val.shape[1]

    model = PredictionModel(input_size)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    predictions, test_loss = test_model(X_val, Y_val, model, loss_fun)
    predictions = predictions.detach().numpy()
    plot_predictions(predictions=predictions)

    sample_input = X_val[0].unsqueeze(0).clone().detach()
    writer.add_graph(model, sample_input)
    writer.close()

#run_model()
#cross_validation()
#run_loaded_model("models/model_v7_p_0.0001_20000_AdamW_testLoss_2.093198776245117.pth")

epochs_list = [1000, 10000, 15000, 20000]
lr_list = [0.01, 0.001, 0.0001, 0.00001]
optimizer_args = ["Adam", "Adagrad", "RMSprop", "AdamW", "SGD"]

for epochs in epochs_list:
    for lr in lr_list:
        for opt_arg in optimizer_args:
            input_size = X_train.shape[1]
            model = PredictionModel(input_size)
            optimizer = select_optimizer(model, opt_arg, lr=lr)
            print(f"Training with epochs={epochs}, lr={lr}, optimizer={opt_arg}")
            run_model(X_train,y_train,X_test,y_test,num_epochs=epochs,lr=lr, opt_func=opt_arg)