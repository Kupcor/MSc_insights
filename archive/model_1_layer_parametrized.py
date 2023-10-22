'''
17.10.2023 model_1_layer

Notes:
Lets step back to the first model. This will be very basic model.

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
import hyper_parameters as hp

#   ____________________________    Prediction Model  _________________________________
'''
Main Architecture 
Test model with one hidden layer
Inputs data will be normalized by using build-in pyTorch batchnormalization
Outputs data will remain same format

Some additional observation and remarks
- Batch normalization improve model outputs a lot
'''                                        
class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=13):
        super().__init__()
        
        self.neuron_num = hidden_layer_size
        self.model_name = "model_1_layers"

        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.bn_input = nn.BatchNorm1d(hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, 1)
        self.dropout = nn.Dropout(0.1)

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.soft = nn.Softplus()
        self.l_relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        result = self.input_layer(x)

        result = self.bn_input(result)

        result = self.elu(result)

        #result = self.dropout(result)

        result = self.output_layer(result)

        return result

#   ____________________________    Training Function  _________________________________
#   Main trainig function
def train_model(model, train_loader, test_loader, loss_fun, opt_func, epochs, X_tensor, y_tensor, train_model_file_name = hp.MODEL_FILE_NAME):
    #writer = SummaryWriter(f"{hp.TENSOR_BOARD_DIR}{train_model_file_name}")
    
    model.train()
    model.to(hp.device)

    scheduler = ls.StepLR(opt_func, step_size=hp.num_epochs/2, gamma=0.9)   #LS scheduler -> TODO parametrized and improve it   
    loss_array = []
    print(test_loader)
    #with open(f'test_data_losses/{model.model_name}_{train_model_file_name}_loss_data.txt', "w") as file:
    start_time = time.time()
    for epoch in range(epochs + 1):
        epoch_loss_accumulator = 0.0
        #for batch_X, batch_y in train_loader:
        #    batch_X = batch_X.to(hp.device)
        #    batch_y = batch_y.to(hp.device)
        X_tensor = X_tensor.to(hp.device)
        y_tensor = y_tensor.to(hp.device)
            
        opt_func.zero_grad()
        predictions = model(X_tensor)
            
        loss = loss_fun(predictions, y_tensor.unsqueeze(1))
        loss.backward()
        opt_func.step()

        epoch_loss_accumulator = loss.item()
            
        average_epoch_loss = epoch_loss_accumulator
            
        if epoch % 100 == 0 or epoch == epochs:
            print(f'Epoch num: {epoch} / {epochs} completed | Loss for this epoch: {average_epoch_loss} | LR: {opt_func.param_groups[0]["lr"]} | Optimizer: {opt_func.__class__.__name__} | Device: {next(model.parameters()).device.type}')
        #    file.write(f'Epoch num: {epoch} / {epochs} completed | Loss for this epoch: {loss.item()} | Current learning rate: {opt_func.param_groups[0]["lr"]}\n')
        
        loss_array.append(average_epoch_loss)
        scheduler.step()
                
        #with torch.no_grad():
        #    test_predictions, real_results, loss_value, input_data, losses = test_model(model, test_loader, loss_fun, show_results=False)
        #    model.train()
                
            #   TODO early stop
            
            #   Write to logs | Tensor Board
            #writer.add_scalar("Loss/Train", average_epoch_loss, epoch)
            #writer.add_scalar("Loss/Validation", loss_value, epoch)

        end_time = time.time()
        train_time = end_time - start_time
        #file.write(f'Train time: {train_time}\n')

    #with open(f'test_data_predictions/{model.model_name}_{train_model_file_name}_test_data.txt', "w") as file:
    #    file.write(f'Results in last iteration:\n')
    #    for i in range(len(predictions)):
    #        formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), batch_y[i].item())
    #        file.write(formatted_line)

    print(f'Training finished. Train time: {train_time}')
    #writer.close()
    return loss_array
    
#   Main test function
def test_model(model, test_loader, X_test, y_test, loss_fun, show_results=True, test_model_file_name=hp.MODEL_FILE_NAME ):
    model.eval()
    model.to(hp.device)
    
    test_loss = 0.0
    losses = []  
    all_predictions = []
    all_reall_results = []

    with torch.no_grad():
        #for batch_X, batch_y in test_loader:
            #batch_X = batch_X.to(hp.device)
            #batch_y = batch_y.to(hp.device)
            X_test = X_test.to(hp.device)
            y_test = y_test.to(hp.device)
            
            predictions = model(X_test)
            predictions = predictions.squeeze(1)

            batch_loss = loss_fun(predictions, y_test)
            test_loss += batch_loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_reall_results.append(y_test.cpu().numpy())
            
            for i in range(len(y_test)):
                single_prediction = predictions[i].cpu().numpy()
                single_target = y_test[i].cpu().numpy()
                single_input = X_test[i].cpu().numpy()
                
                # Oblicz wartość funkcji straty dla pojedynczej predykcji
                single_loss = loss_fun(predictions[i].unsqueeze(0), y_test[i].unsqueeze(0))
                losses.append(single_loss)
        
    average_test_loss = test_loss    #   It is redundant but lets leave it for now

    if show_results:
        print(f'Loss during test: {average_test_loss}')
        print(f'Test finished')
        with open(f'one_layer_results/{model.model_name}_{test_model_file_name}_neuron_number_{model.neuron_num}_{test_loss}_test_results_data.txt', "w") as file:
            file.write(f'Results in last iteration:\n')
            for i in range(len(predictions)):
                formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), y_test[i].item())
                file.write(formatted_line)

    all_reall_results_flat = [item for sublist in all_reall_results for item in sublist]
    return all_predictions, all_reall_results_flat, average_test_loss, X_test, losses