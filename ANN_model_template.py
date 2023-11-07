# Created 21.10.2023

#   Standard python libraries
import time
from datetime import date

#   PyTorch modules
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as ls

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

#   Tensor board
from torch.utils.tensorboard import SummaryWriter

#   Custom functions
import hyper_parameters as hp

#   ____________________________    Prediction Model  _________________________________
'''
Main Architecture 
#   Softplus work the best
#   ELU + Softplus works really goodcls
'''                                    
hidden_layer_activation = nn.ELU()
output_activation = nn.Softplus()

class PredictionModel(nn.Module):
    def __init__(self, hidden_layers_neurons, hidden_layers_activation_function="ReLU", input_size = 10, is_batch_normalization_implemented = True, is_dropout = False, dropout_num = 0.2):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.input_layer = nn.Linear(int(input_size), hidden_layers_neurons[0])
        self.hidden_layers.append(self.input_layer)

        if is_batch_normalization_implemented:
            self.hidden_layers.append(nn.BatchNorm1d(hidden_layers_neurons[0]))
        
        self.hidden_layers.append(hidden_layer_activation)
        
        for i in range(1, len(hidden_layers_neurons)):
            self.hidden_layers.append(nn.Linear(hidden_layers_neurons[i - 1], hidden_layers_neurons[i]))
            
            if is_batch_normalization_implemented:
                self.hidden_layers.append(nn.BatchNorm1d(hidden_layers_neurons[i]))
            
            self.hidden_layers.append(hidden_layer_activation)

            if is_dropout:
                self.hidden_layers.append(nn.Dropout(p=dropout_num))

        self.output_layer = nn.Linear(hidden_layers_neurons[len(hidden_layers_neurons) - 1], 1)
        self.hidden_layers.append(self.output_layer)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = output_activation(x)
        return x

#   ____________________________    Training Functions  _________________________________
#   Basic training function
def train_model(model, X_train, y_train, loss_fun, opt_func, epochs):
    #   Additional data gathering
    losses_for_training_curve = []

    model.train()
    model.to(hp.device)
    X_train = X_train.to(hp.device)
    y_train = y_train.to(hp.device)

    scheduler = ls.StepLR(opt_func, step_size=hp.num_epochs//4, gamma=0.1)   #LS scheduler -> TODO parametrized and improve it   

    start_time = time.time()
    previous_train_loss = 0
    epoch_till_not_change = 0

    for epoch in range(epochs + 1):

        opt_func.zero_grad()
        predictions = model(X_train)
                
        train_loss = loss_fun(predictions, y_train.unsqueeze(1))
        train_loss.backward()
        opt_func.step()
                   
        if epoch % 100 == 0 or epoch == epochs:
            print(f'Epoch num: {epoch} / {epochs} completed | Loss for this epoch: {train_loss.item()} | LR: {round(opt_func.param_groups[0]["lr"], 7)} | Optimizer: {opt_func.__class__.__name__} | Device: {next(model.parameters()).device.type}')
        if epoch < (epochs * 3) // 4:
            scheduler.step()

        #   Additional data gathering
        losses_for_training_curve.append(train_loss.item())
        last_epoch_loss = train_loss.item()

        if abs(train_loss.item() - previous_train_loss) < 0.0001:
            epoch_till_not_change+=1
        else:
            epoch_till_not_change = 0
        if epoch_till_not_change > 100:
            print(f'Early stop in epoch: {epoch}. Loss: {train_loss.item()}')
            previous_train_loss = train_loss.item()
            break

        previous_train_loss = train_loss.item()
        
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Loss in last epoch: {epoch}. Loss: {train_loss.item()}')
    print(f'Training finished. Train time: {train_time}')

    return losses_for_training_curve, predictions, last_epoch_loss
    
#   Basic test function
def test_model(model, X_test, y_test, loss_fun):
    model.eval()

    model.to(hp.device)
    X_test = X_test.to(hp.device)
    y_test = y_test.to(hp.device)

    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.squeeze(1)
        test_loss = loss_fun(predictions, y_test)
    r2 = r2_score(y_test, predictions)
    print(f'\nTest finished')
    print(f'Loss during test: {test_loss.item()}')
    print(f"R root error: {r2}")

    return test_loss.item(), r2

#   Basic validation function
def validate_without_batches(model, X_validate, y_validate, loss_fun, device, threshold=2):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = len(X_validate)

    with torch.no_grad():
        for data, target in zip(X_validate, y_validate):
            data, target = data.to(device), target.to(device)
            
            target = target.unsqueeze(0).unsqueeze(0)
            data = data.unsqueeze(0)

            predictions = model(data)
            loss = loss_fun(predictions, target)
            total_loss += loss.item()

            _, predicted = torch.max(predictions, 1)
            if abs(predicted - target) <= threshold:
                correct_predictions += 1

    accuracy = correct_predictions / total_samples
    average_loss = total_loss / total_samples

    print(f'\nValidation:\nAccuracy: {accuracy}\nAverage loss: {average_loss}')
    model.train()

    return accuracy, average_loss    