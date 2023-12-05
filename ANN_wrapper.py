import ANN_model_template as model
import archive.model_5_layers as ob_model

from datetime import date

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from torch.utils.tensorboard import SummaryWriter

import data_preparation as dp
import hyper_parameters as hp
import snippets as sp
import prepare_outputs as po
import data_preparation as dp

import numpy as np
import pandas as pd

#   ____________________________    Load Data  _________________________________
#   Standard split (train and test data) - it is used in base training (without cross validation)
X, y, hp.x_scaler, hp.y_scaler = dp.get_standarized_data(hp.DATA_FILE)
#X, y = dp.load_training_data(hp.DATA_FILE)
X_train, X_test, y_train, y_test = dp.get_splited_training_data(X, y, hp.SEED, hp.train_size_rate)

def train_model_wrapper(X_train = X_train, y_train = y_train, X_test=X_test, y_test=y_test, X_val=None, y_val=None, hidden_layers_neurons = hp.neurons_in_hidden_layers, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg, model_save=True, show_output_data=True):
    prediction_model = model.PredictionModel(hidden_layers_neurons = hidden_layers_neurons)
    prediction_model = prediction_model.to(hp.device)
    loss_function = nn.MSELoss() 
    
    #   Select optimizer
    optimizer = sp.select_optimizer(prediction_model, opt_arg=opt_func, lr=learning_rate)
    #   Traing model
    losses_for_training_curve, training_predictions, last_epoch_loss = model.train_model(model=prediction_model, X_train=X_train, y_train=y_train, loss_fun=loss_function, opt_func=optimizer, epochs=num_epochs)
    #   Test model
    test_loss = model.test_model(model=prediction_model, X_test=X_test, y_test=y_test, loss_fun=loss_function)
    
    #   Model validation
    if X_val is not None and y_val is not None:
        accuracy, average_loss = model.validate_without_batches(model=prediction_model, X_validate=X_val, y_validate=y_val, loss_fun=loss_function, device = hp.device)

    #   Show plots
    if show_output_data:
        po.plot_predictions(target_data=y_train, loss=last_epoch_loss, predictions=training_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
        po.loss_oscilation(losses=losses_for_training_curve, opt=opt_func, epochs=num_epochs, lr=learning_rate)

    #   Save model
    if model_save:
        sp.save_model(model=prediction_model, hidden_layers_neurons=hidden_layers_neurons, learning_rate=learning_rate, num_epochs=num_epochs, optimizer=optimizer, test_loss=test_loss)
    return test_loss

def bulk_training():
    for i in range(1,10):
        z = [i, i, i]
        print(z)
        train_model_wrapper(hidden_layers_neurons=z, show_output_data=False)

def train_model_with_cross_validation(X_cr = X, y_cr = y, hidden_layers_neurons = hp.neurons_in_hidden_layers, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg):
    #   Cross validation
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_loss_average = 0
    root_average = 0

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_cr, y_cr)):
        print(f"Fold number {fold} start")
        if max(train_idx) >= len(X_cr) or max(test_idx) >= len(X_cr):
            print("Invalid indices in fold", fold)
        else:
            X_train, y_train = X_cr[train_idx], y_cr[train_idx]
            X_test, y_test = X_cr[test_idx], y_cr[test_idx]
        X_tensor_train, y_tensor_train = dp.get_test_data_converted_to_tensor(X_train, y_train)
        X_tensor_test, y_tensor_test = dp.get_test_data_converted_to_tensor(X_test, y_test)
        
        # Train and evaluate the model for this fold.
        test_loss, root = train_model_wrapper(X_train = X_tensor_train, y_train = y_tensor_train, X_test=X_tensor_test, y_test=y_tensor_test, X_val=None, y_val=None, hidden_layers_neurons = hidden_layers_neurons, num_epochs=num_epochs, learning_rate=learning_rate, opt_func=opt_func, model_save=False, show_output_data=False)
        print(f'Finished fold {fold} Test Loss: {test_loss}')
        fold_loss_average += test_loss
        root_average += root
    
    print(f"Average loss during training: {fold_loss_average / num_folds} | Average R root error: {root_average / num_folds}")


#   Second parameter is a file with new predictions
def bulk_predictions(model_file_name, data_file_name="data/bulks.xlsx", standarized_data=False, scaler=None):
    import re
    from openpyxl import load_workbook
    sp.data_reader(data_file_name)

    pattern = r'hidden_layers_(.*?)_'
    neurons = re.search(pattern, model_file_name).group(1).split("-")
    hidden_layers_neuron = [int(neuron) for neuron in neurons]

    saved_model_path = f'trained_models/{model_file_name}'
    prediction_model = model.PredictionModel(hidden_layers_neuron)
    prediction_model.load_state_dict(torch.load(saved_model_path))
    prediction_model = prediction_model.to(hp.device)
    prediction_model.eval()

    data_frames_dict = pd.read_excel(data_file_name, sheet_name=None, engine='openpyxl')
    for sheet_name, data in data_frames_dict.items():
        print(f'Sheet name: {sheet_name}')
        data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
        data.dropna(subset=['Mass Change [mg.cm2]'], inplace=True)
        X = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']].values
        y = data['Mass Change [mg.cm2]'].values
        time = data['Time [h]']

        if len(X) < 1:
            continue

        if standarized_data:
            X = scaler.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        raw_predictions = prediction_model(X_tensor)
            
        data['Mass Change - predictions'] = raw_predictions.detach().numpy()
        predictions = raw_predictions.detach().numpy()
        predictions_list = [item for sublist in predictions for item in sublist]
        sp.save_results_to_excel(predictions_list, data_file_name, sheet_name, sheet_name, time, y)

def bulk_predictions_on_new_data(model_file_name, data_file_name="data/wynikowy_plik_excel.xlsx", standarized_data=False, scaler=None):
    import re
    from openpyxl import load_workbook
    sp.redundant_func(data_file_name)

    pattern = r'hidden_layers_(.*?)_'
    neurons = re.search(pattern, model_file_name).group(1).split("-")
    hidden_layers_neuron = [int(neuron) for neuron in neurons]

    saved_model_path = f'trained_models/{model_file_name}'
    prediction_model = model.PredictionModel(hidden_layers_neuron)
    prediction_model.load_state_dict(torch.load(saved_model_path))
    prediction_model = prediction_model.to(hp.device)
    prediction_model.eval()

    data_frames_dict = pd.read_excel(data_file_name, sheet_name=None, engine='openpyxl')
    for sheet_name, data in data_frames_dict.items():
        print(f'Sheet name: {sheet_name}')
        X = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']].values
        time = data['Time [h]']

        if standarized_data:
            X = scaler.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        raw_predictions = prediction_model(X_tensor)
            
        data['Mass Change - predictions'] = raw_predictions.detach().numpy()
        predictions = raw_predictions.detach().numpy()
        predictions_list = [item for sublist in predictions for item in sublist]
        sp.save_results_to_excel_new_data(predictions_list, data_file_name, sheet_name, sheet_name, time)

#   TODO Leave it for now
def hyper_parameter_training():
    import hyper_parameter_tuning as hpt
    hyperparameters_set = hpt.generate_hyperparameter_combinations(100)
    for hyperparameters in hyperparameters_set:
        hidden_layers_neurons = hyperparameters['hidden_layers_neurons']
        num_epochs = hyperparameters['num_epochs']
        learning_rate = hyperparameters['learning_rate']
        opt_func = hyperparameters['opt_func']

        print(f"Training: {hidden_layers_neurons} {num_epochs} {learning_rate} {opt_func}")

        train_model_wrapper(
            X_train=X_train,
            y_train=y_train,
            hidden_layers_neurons=hidden_layers_neurons,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            opt_func=opt_func,
            model_save=True, 
            show_output_data=False
        )

def load_trained_model(data_file_name="data/chart_data.xlsx", compare_data_file_name="data/data.xlsx"):
    import tkinter as tk
    from tkinter import filedialog
    import re
    from torch.utils.tensorboard import SummaryWriter

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir=r"C:\Users\Piotr Kupczyk\Mój folder\Studia\Informatyczna techniczna\Praca magisterska\Model\trained_models")
    if file_path:
        file_name = file_path.split("/")[-1]
    else:
        return None
    
    pattern = r'hidden_layers_(.*?)_'
    neurons = re.search(pattern, file_name).group(1).split("-")
    hidden_layers_neuron = [int(neuron) for neuron in neurons]

    saved_model_path = f'trained_models/{file_name}'
    prediction_model = model.PredictionModel(hidden_layers_neuron)
    prediction_model.load_state_dict(torch.load(saved_model_path))
    prediction_model = prediction_model.to(hp.device)
    prediction_model.eval()

    writer = SummaryWriter('runs/trained_model_visualization')

    # Dodanie grafu modelu do TensorBoard (potrzebny przykładowy input)
    dummy_input = torch.rand(1, 10)  # Zastąp `appropriate_input_shape` właściwymi wymiarami
    writer.add_graph(prediction_model, dummy_input)

    test_data = dp.get_only_test_data_as_a_tensor(file_name=data_file_name)
    time = np.array(test_data.to('cpu'))[:,-1]

    #test_data = dp.input_normalization_mean(test_data)
    raw_predictions = prediction_model(test_data)

    predictions = raw_predictions.to('cpu').detach().numpy()
    predictions_filtered = [prediction[0] for prediction in predictions]

    if compare_data_file_name is None:
        po.create_graph_of_material_change_over_time(time=time, material=predictions_filtered)
    else:
        compare_time, compare_predictions = dp.get_time_and_mass_change(file_name=compare_data_file_name)
        po.create_graph_of_material_change_over_time(time=time, material=predictions_filtered, time_ref=compare_time, material_ref=compare_predictions)
    
    writer.close()