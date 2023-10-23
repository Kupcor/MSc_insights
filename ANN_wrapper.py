import ANN_model_template as model

from datetime import date

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import data_preparation as dp
import hyper_parameters as hp
import snippets as sp
import prepare_outputs as po

import numpy as np
import pandas as pd

#   ____________________________    Load Data  _________________________________
#   Standard split (train and test data) - it is used in base training (without cross validation)
X_train, X_test, X_validation, y_train, y_test, y_validation = dp.get_splitted_training_test_and_validation_data(hp.DATA_FILE, hp.SEED, hp.train_size_rate)

def train_model_wrapper(X_train = X_train, y_train = y_train, hidden_layers_neurons = hp.neurons_in_hidden_layers, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg, model_save=True, show_output_data=True):
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
    accuracy, average_loss = model.validate_without_batches(model=prediction_model, X_validate=X_validation, y_validate=y_validation, loss_fun=loss_function, device = hp.device)

    #   Show plots
    if show_output_data:
        po.plot_predictions(target_data=y_train, loss=last_epoch_loss, predictions=training_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
        po.loss_oscilation(losses=losses_for_training_curve, opt=opt_func, epochs=num_epochs, lr=learning_rate)

    #   Save model
    if model_save and test_loss < 10:
        sp.save_model(model=prediction_model, hidden_layers_neurons=hidden_layers_neurons, learning_rate=learning_rate, num_epochs=num_epochs, optimizer=optimizer, accuracy=accuracy, average_loss=average_loss, test_loss=test_loss)

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


def load_trained_model(data_file_name="data/chart_data.xlsx", compare_data_file_name="data/rdata.xlsx"):
    import tkinter as tk
    from tkinter import filedialog
    import re

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

def bulk_predictions(data_file_name="data/bulks.xlsx"):
    import tkinter as tk
    from tkinter import filedialog
    import re
    from openpyxl import load_workbook
    sp.data_reader(data_file_name)

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir=r"C:\Users\Piotr Kupczyk\Mój folder\Studia\Informatyczna techniczna\Praca magisterska\Model\trained_models")
    if file_path:
        file_name = file_path.split("/")[-1]
    else:
        return None
    
    print(f"{file_path}")
    
    pattern = r'hidden_layers_(.*?)_'
    neurons = re.search(pattern, file_name).group(1).split("-")
    hidden_layers_neuron = [int(neuron) for neuron in neurons]

    saved_model_path = f'trained_models/{file_name}'
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
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        raw_predictions = prediction_model(X_tensor)
            
        data['Mass Change - predictions'] = raw_predictions.detach().numpy()
        predictions = raw_predictions.detach().numpy()
        predictions_list = [item for sublist in predictions for item in sublist]
        sp.save_results_to_excel(predictions_list, data_file_name, sheet_name, sheet_name, time, y)
