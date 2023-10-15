'''
25.09.2023 model_v11

Notes:
Files and models reorganizations

Results:
--

TODO
- File reorganization -> its big monolit

Maybe TODO | TOIMPLEMENT:
- Hyperparameter tuning
- Model architecture - testing
- TensorBoard - interesting
- Bayes optimalization
'''

#   Standard python libraries
import time
import sys
from datetime import date

#   PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ls

#   Data operations and visualisations libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#   Tensor board
from torch.utils.tensorboard import SummaryWriter

#   Custom functions
import data_preparation as sfdp
import hyper_parameters as hp
import snippets as sp
import prepare_outputs as po
import data_features_analisys as dfa

#   ____________________________    Prediction Model  _________________________________
input_layer_neurons = 13
first_layer_neurons = 15
output_layer_neurons = 4

'''
Main Architecture 
Maybe it is good idea to split it into another file -> TODO in the future
For now it has 5 hidden layer. I do not know yet if it is too small or to much, but for now works fine
I am also using batch normalization | Works fine, better than my implementation in data_preparation.py
For now using ReLU - probably I won't change it
Inputs data will be normalized by using build-in pyTorch batchnormalization
Outputs data will remain same format
'''                                        
class PredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model_name = "model_1_layers"
        self.input_layer = nn.Linear(input_size, input_layer_neurons)
        self.bn_input = nn.BatchNorm1d(input_layer_neurons)
        #self.first_hidden_layer = nn.Linear(input_layer_neurons, output_layer_neurons)
        #self.bn_first = nn.BatchNorm1d(output_layer_neurons)
        #self.second_hidden_layer = nn.Linear(first_layer_neurons, output_layer_neurons)
        #self.bn_second = nn.BatchNorm1d(output_layer_neurons)
        #self.third_hidden_layer = nn.Linear(second_layer_neurons, third_layer_neurons)
        #self.bn_third = nn.BatchNorm1d(third_layer_neurons)
        #self.fourth_hidden_layer = nn.Linear(third_layer_neurons, fourth_layer_neurons)
        #self.bn_fourth = nn.BatchNorm1d(fourth_layer_neurons)
        #self.fifth_hidden_layer = nn.Linear(fourth_layer_neurons, output_layer_neurons)
        self.output_layer = nn.Linear(input_layer_neurons, 1)
        self.relu = nn.ReLU()
        #self.l_relu = nn.LeakyReLU()

    def forward(self, x):
        # Start
        result = self.input_layer(x)
        result = self.bn_input(result)
        result = self.relu(result)
        # 1
        #result = self.first_hidden_layer(result)
        #result = self.bn_first(result)
        #result = self.relu(result)
        #result = self.l_relu(result)
        # 2
        #result = self.second_hidden_layer(result)
        #result = self.relu(result)
        # End
        result = self.output_layer(result)
        return result

#   ____________________________    Training Function  _________________________________
#   Main trainig function
def train_model(model, train_loader, test_loader, loss_fun, opt_func, epochs, train_model_file_name = hp.MODEL_FILE_NAME):
    writer = SummaryWriter(f"{hp.TENSOR_BOARD_DIR}{train_model_file_name}")
    
    model.train()
    model.to(hp.device)

    scheduler = ls.StepLR(opt_func, step_size=hp.num_epochs/2, gamma=0.9)   #LS scheduler -> TODO parametrized and improve it   
    loss_array = []

    with open(f'test_data_losses/{model.model_name}_{train_model_file_name}_loss_data.txt', "w") as file:
        start_time = time.time()
        for epoch in range(epochs + 1):
            epoch_loss_accumulator = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(hp.device)
                batch_y = batch_y.to(hp.device)

                predictions = model(batch_X)
                opt_func.zero_grad()
                
                loss = loss_fun(predictions, batch_y.unsqueeze(1))
                loss.backward()
                opt_func.step()

                epoch_loss_accumulator += loss.item()
            
            average_epoch_loss = epoch_loss_accumulator / len(train_loader)
            
            if epoch % 100 == 0 or epoch == epochs:
                print(f'Epoch num: {epoch} / {epochs} completed | Loss for this epoch: {average_epoch_loss} | LR: {opt_func.param_groups[0]["lr"]} | Optimizer: {opt_func.__class__.__name__} | Device: {next(model.parameters()).device.type}')
                file.write(f'Epoch num: {epoch} / {epochs} completed | Loss for this epoch: {loss.item()} | Current learning rate: {opt_func.param_groups[0]["lr"]}\n')
        
            loss_array.append(average_epoch_loss)
            scheduler.step()
                
            with torch.no_grad():
                test_predictions, real_results, loss_value, input_data, losses = test_model(model, test_loader, loss_fun, show_results=False)
                model.train()
                
            #   TODO early stop
            
            #   Write to logs | Tensor Board
            writer.add_scalar("Loss/Train", average_epoch_loss, epoch)
            writer.add_scalar("Loss/Validation", loss_value, epoch)

        end_time = time.time()
        train_time = end_time - start_time
        file.write(f'Train time: {train_time}\n')

    with open(f'test_data_predictions/{model.model_name}_{train_model_file_name}_test_data.txt', "w") as file:
        file.write(f'Results in last iteration:\n')
        for i in range(len(predictions)):
            formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), batch_y[i].item())
            file.write(formatted_line)

    print(f'Training finished. Train time: {train_time}')
    writer.close()
    return loss_array
    
#   Main test function
def test_model(model, test_loader, loss_fun, show_results=True, test_model_file_name=hp.MODEL_FILE_NAME ):
    model.eval()
    model.to(hp.device)
    
    test_loss = 0.0
    losses = []  
    all_predictions = []
    all_reall_results = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(hp.device)
            batch_y = batch_y.to(hp.device)
            
            predictions = model(batch_X)
            predictions = predictions.squeeze(1)

            batch_loss = loss_fun(predictions, batch_y)
            test_loss += batch_loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_reall_results.append(batch_y.cpu().numpy())
            
            for i in range(len(batch_y)):
                single_prediction = predictions[i].cpu().numpy()
                single_target = batch_y[i].cpu().numpy()
                single_input = batch_X[i].cpu().numpy()
                
                # Oblicz wartość funkcji straty dla pojedynczej predykcji
                single_loss = loss_fun(predictions[i].unsqueeze(0), batch_y[i].unsqueeze(0))
                losses.append(single_loss)
        
    average_test_loss = test_loss / len(test_loader)    #   It is redundant but lets leave it for now

    if show_results:
        print(f'Loss during test: {average_test_loss}')
        print(f'Test finished')
        with open(f'test_data_run/{model.model_name}_{test_model_file_name}_test_results_data.txt', "w") as file:
            file.write(f'Results in last iteration:\n')
            for i in range(len(predictions)):
                formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), batch_y[i].item())
                file.write(formatted_line)

    all_reall_results_flat = [item for sublist in all_reall_results for item in sublist]
    return all_predictions, all_reall_results_flat, average_test_loss, batch_X, losses