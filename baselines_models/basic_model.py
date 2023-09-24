import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

import sub_func.data_preparation as sfdp

#   19.08.2023
#   Pre data

MODEL_NAME = "basic_model"

train_size_rate = 0.8
lr = 0.01
num_epochs = 1000
dropout_number = 0.15

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

# Funconts and classes
class PredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        result = self.input_layer(x)
        result = self.output_layer(x)
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
                print(f'Epoch num: {epoch} / {epochs} completed/ | Loss for this epoch: {loss.item()}')
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
    with open(MODEL_NAME + "_test_results_data.txt", "w") as file:
        file.write(f'Results in last iteration:\n')
        for i in range(len(predictions)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(
                i, predictions[i].item(), y[i].item()
            )
            file.write(formatted_line)
    print(f'Loss during test: {loss.item()}')
    print(f'Test finished')

# Work

X_tensor, y_tensor = sfdp.get_prepared_data("data.xlsx")

train_size = int(train_size_rate * len(X_tensor))
test_size = len(X_tensor) - train_size

X_train, X_test = torch.split(X_tensor, [train_size, test_size])
y_train, y_test = torch.split(y_tensor, [train_size, test_size])

input_size = X_train.shape[1]
Prediction_model_v = PredictionModel(input_size)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(Prediction_model_v.parameters(), lr) 

train_model(X_train, y_train, Prediction_model_v, loss_fn, optimizer, num_epochs)
test_model(X_test, y_test, Prediction_model_v, loss_fn)


