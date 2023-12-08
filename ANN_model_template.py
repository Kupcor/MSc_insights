#   Standard python libraries
import time
from datetime import date

#   PyTorch modules
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as ls
import torch.nn.init as init

#   sklearn - machine learning lib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

#   Tensor board
from torch.utils.tensorboard import SummaryWriter

"""
Main architecture of prediction model
"""   
class PredictionModel(nn.Module):
    def __init__(self, hidden_layers_neurons, input_size=10, num_epochs=1000, hidden_layers_activation_function="LeakyReLU",
                 is_batch_normalization_implemented=False, is_dropout=False, dropout_num=0.1, weight_init_method='xavier_uniform_'):
        
        super(PredictionModel, self).__init__()

        self.weight_init_method = weight_init_method

        #   Set input layer
        self.input_layer = nn.Linear(int(input_size), hidden_layers_neurons[0])
        #   Set first hidden layer
        self.hidden_layers = nn.ModuleList([self.input_layer])

        # Initialize weights of the input layer
        self.weight_init(self.input_layer.weight)

        # Add batch normalization if specified
        if is_batch_normalization_implemented:
            self.hidden_layers.append(nn.BatchNorm1d(hidden_layers_neurons[0]))
        
        #   Add activation function to first hidden layer
        self.hidden_layers.append(getattr(nn, hidden_layers_activation_function)())
        
        # Define hidden layers
        for i in range(1, len(hidden_layers_neurons)):
            hidden_layer = nn.Linear(hidden_layers_neurons[i - 1], hidden_layers_neurons[i])
            self.hidden_layers.append(hidden_layer)
            
            # Initialize weights of the hidden layer
            self.weight_init(hidden_layer.weight)

            if is_batch_normalization_implemented:
                self.hidden_layers.append(nn.BatchNorm1d(hidden_layers_neurons[i]))
            
            #   Set activation function to each hidden layer
            self.hidden_layers.append(getattr(nn, hidden_layers_activation_function)())

            if is_dropout:
                self.hidden_layers.append(nn.Dropout(p=dropout_num))

        #   Set output layer
        self.output_layer = nn.Linear(hidden_layers_neurons[-1], 1)
        self.hidden_layers.append(self.output_layer)

        self.weight_init(self.output_layer.weight)
    
    def weight_init(self, weight):
        if self.weight_init_method == 'xavier_uniform_':
            nn.init.xavier_uniform_(weight)
        elif self.weight_init_method == 'kaiming_normal_':
            nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def fit(self, X_train, y_train, loss_fun, opt_func, epochs, device):
        self.train()
        self.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        scheduler = ls.StepLR(opt_func, step_size=epochs//3, gamma=0.1)   #LS scheduler -> TODO parametrized and improve it   

        start_time = time.time()
        previous_train_loss = 0
        epoch_till_not_change = 0

        for epoch in range(epochs + 1):

            opt_func.zero_grad()
            predictions = self(X_train)
                    
            train_loss = loss_fun(predictions, y_train.unsqueeze(1))
            train_loss.backward()
            opt_func.step()
                    
            if epoch % 100 == 0 or epoch == epochs:
                print(f'Epoch num: {epoch} / {epochs} completed | Loss for this epoch: {train_loss.item()} | LR: {round(opt_func.param_groups[0]["lr"], 7)} | Optimizer: {opt_func.__class__.__name__} | Device: {next(model.parameters()).device.type}')
            
            scheduler.step()
            
        end_time = time.time()
        train_time = end_time - start_time
        print(f'Loss in last epoch: {epoch}. Loss: {train_loss.item()}')
        print(f'Training finished. Train time: {train_time}')

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

"""
Training function
Adjusting model weights
Gathering training model data
"""
def train_model(model, X_train, y_train, loss_fun, opt_func, epochs, device):
    #   Additional data gathering
    losses_for_training_curve = []

    model.train()
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    scheduler = ls.StepLR(opt_func, step_size=epochs//3, gamma=0.1)   #LS scheduler -> TODO parametrized and improve it   

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
        
        scheduler.step()

        #   Additional data gathering
        losses_for_training_curve.append(train_loss.item())
        last_epoch_loss = train_loss.item()

        if abs(train_loss.item() - previous_train_loss) < 0.0001:
            epoch_till_not_change+=1
        else:
            epoch_till_not_change=0

        previous_train_loss = train_loss.item()

        if epoch_till_not_change > 300:
            print(f'Early stop in epoch: {epoch}. Loss: {train_loss.item()}')
            break

    end_time = time.time()
    train_time = end_time - start_time
    print(f'Loss in last epoch: {epoch}. Loss: {train_loss.item()}')
    print(f'Training finished. Train time: {train_time}')

    return losses_for_training_curve, predictions, last_epoch_loss
    
"""
Main test function
Validation trained model
"""
def test_model(model, X_test, y_test, loss_fun, device):
    model.eval()

    model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.squeeze(1)
        test_loss = loss_fun(predictions, y_test)
    r2 = r2_score(y_test, predictions)
    print(f'\nTest finished')
    print(f'Loss during test: {test_loss.item()}')
    print(f"R square: {r2}")

    return test_loss.item(), r2, predictions

"""
Main validation function
TODO -> cross validation to be done
"""
def validate_regression_model(model, X_validate, y_validate, loss_fun, device):
    model.eval()

    total_loss = 0.0
    total_samples = len(X_validate)

    predictions = []
    
    with torch.no_grad():
        for data, target in zip(X_validate, y_validate):
            data, target = data.to(device), target.to(device)
            output = model(data.unsqueeze(0))
            loss = loss_fun(output, target.unsqueeze(0).unsqueeze(0))
            total_loss += loss.item()
            predictions.append(output)

    predictions = torch.cat(predictions, dim=0)

    predictions_np = predictions.cpu().numpy()
    y_validate_np = y_validate.cpu().numpy()
    r2 = r2_score(y_validate_np, predictions_np)
    mse = mean_squared_error(y_validate_np, predictions_np)

    average_loss = total_loss / total_samples

    print(f'\nValidation:\nMSE: {mse}\nAverage loss: {average_loss}\nR2 Score: {r2}\n')
    model.train()

    return mse, average_loss, r2



