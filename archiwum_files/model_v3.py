import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sub_func.data_preparation as sfdp

#   21.08.2023 model_v3
#   And it is preatty good. It return Loss during test on level lower than 10.
#   Pre data

MODEL_NAME = sys.argv[0]

train_size_rate = 0.2
lr = 0.01
num_epochs = 1000
dropout_number = 0.15

X_tensor, y_tensor = sfdp.get_prepared_data("data.xlsx")

X_train, X_test, y_train, y_test = train_test_split(X_tensor, 
                                                    y_tensor, 
                                                    test_size=train_size_rate,
                                                    random_state=42) #  Chce powtarzalności na danych na okres testowania


## TODO:
## Cross validation
## Hyperparameter tuning
## Model architecture - testing
## Normalization
## Learning rate scheduler
## TensorBoard

if len(sys.argv) > 1:
    print(f"Lr set to {sys.argv[1]}")
    lr = float(sys.argv[1]) 

if len(sys.argv) > 2:
    print(f"Epoch number set to {sys.argv[2]}")
    num_epochs = int(sys.argv[2])

#   Architecture 
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

def train_model(X, y, model, loss_fun, opt_func, epochs):
    model.train()
    with open("test_data/"+MODEL_NAME + "_loss_data.txt", "w") as file:
        start_time = time.time()
        for epoch in range(epochs+1):
            predicitons = model(X)
            loss = loss_fun(predicitons, y.unsqueeze(1))
            opt_func.zero_grad()
            loss.backward()
            opt_func.step()
            if epoch % 100 == 0 or epoch == epochs:
                print(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()} | Accuracy: {accuracy_fn(y, predicitons)}')
                file.write(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()}\n')
        end_time = time.time()
        train_time = end_time - start_time
        file.write(f'Train time: {train_time}\n')
    
    with open("test_data/"+MODEL_NAME + "_train_results_data.txt", "w") as file:
        file.write(f'Results in last iteration:\n')
        for i in range(len(predicitons)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(
                i, predicitons[i].item(), y[i].item()
            )
            file.write(formatted_line)
    print(f'Training finished. Train time: {train_time}')

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
    return predictions

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  # plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  info_text = f"File: {MODEL_NAME}\nEpoch: {num_epochs}\nLR: {lr}\nTraining rate: {train_size_rate}"
  plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

  z = list(range(len(test_labels)))
  plt.scatter(z, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    plt.scatter(z, predictions, c="r", s=4, label="Predictions")

  plt.legend(prop={"size": 14})
  plt.grid()
  plt.show()

def run_model(X_tr=X_train, y_tr=y_train, X_ts = X_test, y_ts = y_test):
    input_size = X_tr.shape[1]
    Prediction_model_v = PredictionModel(input_size)

    #loss_fn = nn.L1Loss() # MAE żądzi
    loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(Prediction_model_v.parameters(), lr) 
    #optimizer = torch.optim.SGD(Prediction_model_v.parameters(), lr)

    train_model(X_tr, y_tr, Prediction_model_v, loss_fn, optimizer, num_epochs)
    predictions = test_model(X_ts, y_ts, Prediction_model_v, loss_fn)
    predictions = predictions.detach().numpy()
    #print(predictions)
    plot_predictions(predictions=predictions)

run_model()
