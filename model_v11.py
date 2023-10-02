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
        self.input_layer = nn.Linear(input_size, hp.input_layer_neurons)
        self.bn_input = nn.BatchNorm1d(hp.input_layer_neurons)
        self.first_hidden_layer = nn.Linear(hp.input_layer_neurons, hp.first_layer_neurons)
        self.bn_first = nn.BatchNorm1d(hp.first_layer_neurons)
        self.second_hidden_layer = nn.Linear(hp.first_layer_neurons, hp.second_layer_neurons)
        self.bn_second = nn.BatchNorm1d(hp.second_layer_neurons)
        self.third_hidden_layer = nn.Linear(hp.second_layer_neurons, hp.third_layer_neurons)
        self.bn_third = nn.BatchNorm1d(hp.third_layer_neurons)
        self.fourth_hidden_layer = nn.Linear(hp.third_layer_neurons, hp.fourth_layer_neurons)
        self.bn_fourth = nn.BatchNorm1d(hp.fourth_layer_neurons)
        self.fifth_hidden_layer = nn.Linear(hp.fourth_layer_neurons, hp.output_layer_neurons)
        self.output_layer = nn.Linear(hp.output_layer_neurons, 1)
        self.relu = nn.ReLU()
        #self.l_relu = nn.LeakyReLU()

    def forward(self, x):
        # Start
        result = self.input_layer(x)
        result = self.bn_input(result)
        result = self.relu(result)
        # 1
        result = self.first_hidden_layer(result)
        result = self.bn_first(result)
        result = self.relu(result)
        #result = self.l_relu(result)
        # 2
        result = self.second_hidden_layer(result)
        result = self.bn_second(result)
        result = self.relu(result)
        # 3
        result = self.third_hidden_layer(result)
        result = self.bn_third(result)
        result = self.relu(result)
        # 4
        result = self.fourth_hidden_layer(result)
        result = self.bn_fourth(result)
        result = self.relu(result)
        # 5
        result = self.fifth_hidden_layer(result)
        result = self.relu(result)
        # End
        result = self.output_layer(result)
        return result

#   ____________________________    Training Function  _________________________________
#   Main trainig function
def train_model(model, train_loader, loss_fun, opt_func, epochs, train_model_file_name = hp.MODEL_FILE_NAME):
    writer = SummaryWriter(f"{hp.TENSOR_BOARD_DIR}{train_model_file_name}")
    
    model.train()
    model.to(hp.device)

    scheduler = ls.StepLR(opt_func, step_size=hp.num_epochs/2, gamma=0.9)   #LS scheduler -> TODO parametrized and improve it   
    loss_array = []

    with open(f'test_data_losses/{hp.MODEL_NAME}_{train_model_file_name}_loss_data.txt', "w") as file:
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

    with open(f'test_data_predictions/{hp.MODEL_NAME}_{train_model_file_name}_test_data.txt', "w") as file:
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

            batch_loss = loss_fun(predictions, batch_y.unsqueeze(1))
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
        with open(f'test_data_run/{hp.MODEL_NAME}_{test_model_file_name}_test_results_data.txt', "w") as file:
            file.write(f'Results in last iteration:\n')
            for i in range(len(predictions)):
                formatted_line = '{:<2}. Predicted: {:<10.4f} | Actual: {:<10}\n'.format(i, predictions[i].item(), batch_y[i].item())
                file.write(formatted_line)

    all_reall_results_flat = [item for sublist in all_reall_results for item in sublist]
    return all_predictions, all_reall_results_flat, average_test_loss, batch_X, losses

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


#   ____________________________    Load Data  _________________________________
#   Standard split (train and test data) - it is used in base training (without cross validation)
X_train, X_test, y_train, y_test = sfdp.get_splited_training_data(hp.DATA_FILE, hp.SEED, hp.train_size_rate)
#   Create data sets (in data loaders form)
train_loader, test_loader = sfdp.get_test_and_train_loader(hp.DATA_FILE, hp.SEED, hp.train_size_rate, hp.batch_size, True)
#   Input size
input_size = X_train.shape[1]

#   Standard run
def run_model(train_loader = train_loader, test_loader = test_loader, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg):
    prediction_model = PredictionModel(input_size)
    prediction_model = prediction_model.to(hp.device)
    loss_function = nn.MSELoss() 
    
    #   Select optimizer
    optimizer = sp.select_optimizer(prediction_model, opt_arg=opt_func, lr=learning_rate)

    #   Traing model
    train_loss = train_model(prediction_model, train_loader,  loss_function, optimizer, num_epochs)
    #   Test model
    test_predictions, real_results, test_loss, input_data, losses = test_model(prediction_model, test_loader, loss_function)

    #   Show plots
    po.plot_predictions(test_data=real_results, predictions=test_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
    po.loss_oscilation(train_loss, opt=opt_func, epochs=num_epochs, lr=learning_rate)

    #   Save model
    file_name = f"_model_{learning_rate}_{num_epochs}_{optimizer.__class__.__name__}_{hp.today}_testLoss_{test_loss}"
    save_path = "models/" + hp.MODEL_NAME + file_name + '.pth'
    torch.save(prediction_model.state_dict(), save_path)

def run_loaded_model(file_name, test_loader=test_loader, loss_fun = nn.MSELoss(), X_val = X_test):
    from tensorboardX import SummaryWriter

    tensor_flow_folder = file_name
    import re
    match = re.search(r'\d{4}-\d{2}-\d{2}', tensor_flow_folder)  # Finde date in proper format
    if match:
        date = match.group(0)
        result = tensor_flow_folder[:match.start() + len(date)]
        print(result) 
    else:
        print("Wrong file name")
        exit()

    dict = f"tensor_board_logs/{tensor_flow_folder}"
    writer = SummaryWriter(dict)

    saved_model_path = f'models/{file_name}'
    model = PredictionModel(input_size)
    model.load_state_dict(torch.load(saved_model_path))
    model = model.to(hp.device)

    #dfa.plot_feature_importance(model, X=X_test, feature_names = hp.FEATURES)

    model.eval()
    predictions, real_values, test_loss, input_data, losses = test_model(test_loader=test_loader, model=model, loss_fun=nn.MSELoss(), show_results=True)
    #plot_predictions(test_data=real_values, predictions=predictions)
    # Zapisz dane do pliku
    with open(f'models_results/{file_name}_results.txt', "w") as file:
        headers = ["Index", "Temperature[C]", "Zr[at%]", "Nb[at%]", "Mo[at%]", "Cr[at%]", "Al[at%]", "Ti[at%]", "Ta[at%]", "W[at%]", "Time[h]", "MassChange[mg.cm2](Real)", "MassChange[mg.cm2](Predicted)", "Loss"]
        header_line = "{:<6} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<22} {:<22} {:<22}\n".format(*headers)
        file.write(header_line)

        prd = predictions[0]
        values = input_data

        for i in range(len(prd)):
            values_string = " ".join("{:<15.5f}".format(value) for value in values[i])
            prd_value = "{:<22.5f}".format(prd[i][0])
            real_value = "{:<22.5f}".format(real_values[i])
            loss = "{:<22.5f}".format(losses[i])
            line = "{:<6} {} {} {} {}\n".format(i, values_string, prd_value, real_value, loss)
            file.write(line)
       
    sample_input = X_val[0].unsqueeze(0).clone().detach()
    sample_input = sample_input.to(hp.device)
    writer.add_graph(model, sample_input)
    writer.close()

def run_model_on_test_data():
    X_tensor_2, y_tensor_2 = sfdp.get_test_data_converted_to_tensor(hp.TEST_DATA)
    test_dataset_2 = torch.utils.data.TensorDataset(X_tensor_2, y_tensor_2)
    test_loader_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=hp.batch_size)
    for batch_X, batch_y in test_loader_2:
        print("Batch X:", len(batch_X))
        print("Batch y:", len(batch_y))
    run_loaded_model("Best_models/model_v10_p_0.001_1500_AdamW_testLoss_1.6069554090499878_2023-09-23.pth", "model_v9_p_0.001_3000_Adam_2023-08-30", test_loader=test_loader_2)

def load_model(file_name, test_loader=test_loader, loss_fun = nn.MSELoss(), X_val = X_test):
    from tensorboardX import SummaryWriter

    tensor_flow_folder = file_name
    import re
    match = re.search(r'\d{4}-\d{2}-\d{2}', tensor_flow_folder)  # Finde date in proper format
    if match:
        date = match.group(0)
        result = tensor_flow_folder[:match.start() + len(date)]
        print(result) 
    else:
        print("Wrong file name")
        exit()

    dict = f"tensor_board_logs/{tensor_flow_folder}"
    #writer = SummaryWriter(dict)

    saved_model_path = f'models/{file_name}'
    model = PredictionModel(input_size)
    model.load_state_dict(torch.load(saved_model_path))
    model = model.to(hp.device)

    model.eval()
    X = "1200	0	0	19,5	19,8	20,2	20,4	20,1	0	24"
    X = X.split("\t")
    X_data = []
    for x in X:
        x = x.replace(",",".")
        X_data.append(float(x))
    print(X_data)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    X_tensor = X_tensor.unsqueeze(0)
    X_tensor = X_tensor.to(hp.device)
    predictions = model(X_tensor)
    print(predictions)

    #dfa.plot_feature_importance(model)

#run_model()
#cross_validation()
#run_loaded_model("model_v11_model_0.0001_2000_AdamW_2023-09-28_testLoss_0.8954169750213623.pth", test_loader=train_loader)
#run_model_on_test_data()
