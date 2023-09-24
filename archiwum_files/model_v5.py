import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

import sub_func.data_preparation as sfdp

#   22.08.2023 model_v5
#   Note: Added cross validation in this model version
#   Separeta architecture from file
#   Added lr scheduler
#   No we are using other optimizing methods

#   Notes
#   Lower LR and high epoch give the best effect
#   Best results - lr 1e5 and epoch = 40000, optimizer Adam

#   Results
#   It's hard to obtain lower loss than 0.75 during training

#   Maybe TODO:
#   # Cross validation
#   # Hyperparameter tuning
#   # Model architecture - testing
#   # Learning rate scheduler
#   # TensorBoard - interesting


#   Hyperparameters and Pre-Data
MODEL_NAME = sys.argv[0]
train_size_rate = 0.2
lr = 0.01
num_epochs = 1000
#dropout_number = 0.15

#   Load data to model
X_tensor, y_tensor = sfdp.get_prepared_data("data.xlsx")

#   Standard split (train and tes data)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, 
                                                    y_tensor, 
                                                    test_size=train_size_rate,
                                                    random_state=42) #  Chce powtarzalności na danych na okres testowania


#   Just to have an option to set parameter via console
if len(sys.argv) > 1:
    print(f"Lr set to {sys.argv[1]}")
    lr = float(sys.argv[1]) 

if len(sys.argv) > 2:
    print(f"Epoch number set to {sys.argv[2]}")
    num_epochs = int(sys.argv[2])

#   Main Architecture 
#   Maybe it is good idea to split it into another file
#   For now it has 3 hidden layer. I do not know yet if it is too small or to much, but for now works fine
#   I am also using batch normalization | Works fine, better than my implementation in data_preparation.py
#   For now using ReLU - probably I won't change it
class PredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 128)
        self.bn_input = nn.BatchNorm1d(128)
        self.first_hidden_layer = nn.Linear(128, 256)
        self.bn_first = nn.BatchNorm1d(256)
        self.second_hidden_layer = nn.Linear(256, 128)
        self.bn_second = nn.BatchNorm1d(128)
        self.third_hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        result = self.input_layer(x)
        result = self.bn_input(result)
        result = self.relu(result)
        result = self.first_hidden_layer(result)
        result = self.bn_first(result)
        result = self.relu(result)
        result = self.second_hidden_layer(result)
        result = self.bn_second(result)
        result = self.relu(result)
        result = self.third_hidden_layer(result)
        result = self.relu(result)
        result = self.output_layer(result)
        return result

#   Main trainig function
#   I wanted to retain some 'elasticity'
def train_model(X, y, model, loss_fun, opt_func, epochs):
    model.train()
    
    with open("test_data/" + MODEL_NAME + "_loss_data.txt", "w") as file:
        start_time = time.time()
        for epoch in range(epochs + 1):
            predictions = model(X)
            loss = loss_fun(predictions, y.unsqueeze(1))
            opt_func.zero_grad()
            loss.backward()
            opt_func.step()
            if epoch % 100 == 0 or epoch == epochs:
                print(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()} | LR: {lr}')
                #file.write(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()}\n') #   To speed up things
        end_time = time.time()
        train_time = end_time - start_time
        file.write(f'Train time: {train_time}\n')
    
    with open("test_data/"+MODEL_NAME + "_train_results_data.txt", "w") as file:
        file.write(f'Results in last iteration:\n')
        for i in range(len(predictions)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(
                i, predictions[i].item(), y[i].item()
            )
            file.write(formatted_line)
    print(f'Training finished. Train time: {train_time}')

#   Main test function
def test_model(X,y,model,loss_fun):
    model.eval()
    predictions = model(X)
    loss = loss_fun(predictions, y.unsqueeze(1))
    with open("test_data/"+MODEL_NAME + "_test_results_data.txt", "w") as file:
        file.write(f'Results in last iteration:\n')
        for i in range(len(predictions)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(
                i, predictions[i].item(), y[i].item()
            )
            file.write(formatted_line)
    print(f'Loss during test: {loss.item()}')
    print(f'Test finished')
    return predictions, loss.item()

#   Results comparison in plot
def plot_predictions(train_labels=y_train, 
                     test_labels=y_test, 
                     predictions=None):
  plt.figure(figsize=(15, 7))
  info_text = f"File: {MODEL_NAME}\nEpoch: {num_epochs}\nLR: {lr}\nTraining rate: {train_size_rate}"
  plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

  z = list(range(len(test_labels)))
  plt.scatter(z, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    plt.scatter(z, predictions, c="r", s=4, label="Predictions")

  plt.legend(prop={"size": 14})
  plt.grid()
  plt.show()

#   Cross validation, similar to test model function, but it use cross validation
#   For now I do not want it to be saved in file
def cross_validation(model, optimizer, X=X_tensor, y=y_tensor, loss_fn=nn.MSELoss(), num_epochs=num_epochs, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold_num+1}/{num_folds}")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        input_size = X_train.shape[1]
        model = PredictionModel(input_size)  #  New instance of model for each fold
        
        #   optimizer = optim.Adam(model.parameters(), lr)  #   Again we used Adam
        #   optimizer = optim.Adagrad(model.parameters(), lr)
        #   optimizer = optim.RMSprop(model.parameters(), lr)
        #   optimizer = optim.AdamW(model.parameters(), lr) #   AdamW
        optimizer = optim.LBFGS(model.parameters(), lr)
        
        train_model(X_train, y_train, model, loss_fn, optimizer, num_epochs)
        
        predictions, test_loss = test_model(X_val, y_val, model, loss_fn)
        
        fold_results.append(test_loss)
    print(fold_results)
    #return fold_results

def run_model(X_tr=X_train, y_tr=y_train, X_ts = X_test, y_ts = y_test):
    input_size = X_tr.shape[1]
    Prediction_model_v = PredictionModel(input_size)

    #loss_fn = nn.L1Loss() # MAE żądzi
    loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(Prediction_model_v.parameters(), lr) 
    #optimizer = torch.optim.SGD(Prediction_model_v.parameters(), lr)

    train_model(X_tr, y_tr, Prediction_model_v, loss_fn, optimizer, num_epochs)
    predictions, test_loss = test_model(X_ts, y_ts, Prediction_model_v, loss_fn)
    predictions = predictions.detach().numpy()
    #print(predictions)
    plot_predictions(predictions=predictions)
    save_path = MODEL_NAME + '.pth'
    torch.save(Prediction_model_v.state_dict(), save_path)

def run_model_with_cross_validation():
    input_size = X_tensor.shape[1]
    Prediction_model_v = PredictionModel(input_size)

    #loss_fn = nn.L1Loss() # MAE żądzi
    loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(Prediction_model_v.parameters(), lr) 
    #optimizer = torch.optim.SGD(Prediction_model_v.parameters(), lr)

    cross_validation(Prediction_model_v, optimizer)

run_model()
#run_model_with_cross_validation()