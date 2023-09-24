import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sub_func.data_preparation as sfdp

#   21.08.2023 model_v2
#   Pre data

MODEL_NAME = sys.argv[0]

train_size_rate = 0.05
lr = 0.01
num_epochs = 1000
dropout_number = 0.15

X_tensor, y_tensor = sfdp.get_prepared_data("data.xlsx")

X_train, X_test, y_train, y_test = train_test_split(X_tensor, 
                                                    y_tensor, 
                                                    test_size=train_size_rate, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible


## TODO:
## Cross validation
## Hyperparameter tuning
## Model architecture
## Normalization
## Learning rate scheduler
## TensorBoard

if len(sys.argv) > 1:
    print(f"Lr set to {sys.argv[1]}")
    lr = float(sys.argv[1]) 

if len(sys.argv) > 2:
    print(f"Epoch number set to {sys.argv[2]}")
    num_epochs = int(sys.argv[2])

#   Funconts and classes
#   Architecture 
class PredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        result = self.input_layer(x)
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
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
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
  
  # Plot test data in green
  z = list(range(len(test_labels)))
  plt.scatter(z, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(z, predictions, c="r", s=4, label="Predictions")

  # Show the legend
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
