#   My custo files
import ANN_model_template as model
import data_preparation as dp
import hyper_parameters as hp
import snippets as sp
import prepare_outputs as po
import data_preparation as dp

#   standard python libraries
from datetime import date

#   PyTorch
import torch
import torch.nn as nn

#   Sklearn libraries
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

#   TensorBoard
from torch.utils.tensorboard import SummaryWriter

#   Standard ML and DS libraries
import numpy as np
import pandas as pd

#   ____________________________    Load Data  _________________________________
#   Standard split (train and test data) - it is used in base training (without cross validation)
X, y, hp.x_scaler, hp.y_scaler = dp.get_standarized_data(hp.DATA_FILE)
#X, y = dp.load_training_data(hp.DATA_FILE)
X_train, X_test, y_train, y_test = dp.get_splited_training_data(X, y, hp.SEED, hp.train_size_rate)
input_size = X_train.shape[1]

"""
Training model wrapper
Entire process of training model
Standard model training and testing
"""
def train_model_wrapper(X_train = X_train, y_train = y_train, X_test=X_test, y_test=y_test, X_val=None, y_val=None, hidden_layers_neurons = hp.neurons_in_hidden_layers, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg, model_save=True, show_output_data=True, device=hp.device, input_size=input_size):
    prediction_model = model.PredictionModel(hidden_layers_neurons = hidden_layers_neurons, is_dropout=False, input_size=input_size)
    prediction_model = prediction_model.to(device)
    loss_function = nn.MSELoss() 
    
    #   Select optimizer
    optimizer = sp.select_optimizer(prediction_model, opt_arg=opt_func, lr=learning_rate)
    #   Traing model
    losses_for_training_curve, training_predictions, last_epoch_loss = model.train_model(model=prediction_model, X_train=X_train, y_train=y_train, loss_fun=loss_function, opt_func=optimizer, epochs=num_epochs, device=device)
    #   Test model
    test_loss, r2, test_predictions = model.test_model(model=prediction_model, X_test=X_test, y_test=y_test, loss_fun=loss_function, device=device)
    
    #   Model validation
    if X_val is not None and y_val is not None:
        accuracy, average_loss = model.validate_without_batches(model=prediction_model, X_validate=X_val, y_validate=y_val, loss_fun=loss_function, device = hp.device)

    #   Show plots
    if show_output_data:
        po.plot_predictions(target_data=y_train, loss=last_epoch_loss, predictions=training_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
        po.loss_oscilation(losses=losses_for_training_curve, opt=opt_func, epochs=num_epochs, lr=learning_rate)
        po.scatter_plot(target_data=y_train, loss=last_epoch_loss, predictions=training_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
        po.scatter_plot(target_data=y_test, loss=test_loss, predictions=test_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs, title="Test predictions")

    #   Save model
    if model_save:
        sp.save_model(model=prediction_model, hidden_layers_neurons=hidden_layers_neurons, learning_rate=learning_rate, num_epochs=num_epochs, optimizer=optimizer, test_loss=test_loss, r2=r2)
    return test_loss

"""
Cross validation
To be honest, for now it does not make sense
"""
def cross_validate(X=X, y=y, hidden_layers_neurons=hp.neurons_in_hidden_layers, optimizer_fn=hp.optimizer_arg, lr=hp.lr, epochs=hp.num_epochs, device=hp.device, n_splits=5, threshold=0.5):
    kf = KFold(n_splits=n_splits)
    loss_function = nn.MSELoss() 

    accuracies = []
    losses = []
    r2s = []

    X, y = dp.get_test_data_converted_to_tensor(X, y)
    i = 0

    for train_index, validate_index in kf.split(X):
        print(f"Cross val: {i}")
        i += 1
        X_train, X_validate = X[train_index], X[validate_index]
        y_train, y_validate = y[train_index], y[validate_index]

        prediction_model = model.PredictionModel(hidden_layers_neurons = hidden_layers_neurons, is_dropout=True, input_size=input_size)
        prediction_model = prediction_model.to(device)

        optimizer = sp.select_optimizer(prediction_model, opt_arg=optimizer_fn, lr=lr)

        losses_for_training_curve, training_predictions, last_epoch_loss = model.train_model(model=prediction_model, X_train=X_train, y_train=y_train, loss_fun=loss_function, opt_func=optimizer, epochs=epochs, device=device)

        # Walidacja
        accuracy, average_loss, r2 = model.validate_regression_model(prediction_model, X_validate, y_validate, loss_function, device)
        accuracies.append(accuracy)
        losses.append(average_loss)
        r2s.append(r2)
    print("Summary:")
    print(f"{sum(accuracies) / len(accuracies)}\n{sum(losses) / len(losses)}\n{sum(r2s) / len(r2s)}")
    return sum(accuracies) / len(accuracies), sum(losses) / len(losses)


"""
Bulks predictions
Predictions on individual data sets
"""
def bulk_predictions(model_file_name, data_file_name="data/bulks.xlsx", standarized_data=False, scaler_x=None, scaler_y=None):
    import re
    from openpyxl import load_workbook
    sp.create_file_with_unique_sets(data_file_name)  #   This is very important function here!
    
    print(f"Started operations on file {data_file_name}")

    #   Read necessery information from model_file_name
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
        y_ground_truth = data['Mass Change [mg.cm2]']
        data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
        data.dropna(subset=['Mass Change [mg.cm2]'], inplace=True)
        X = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']].values
        y = data['Mass Change [mg.cm2]'].values
        time = data['Time [h]']

        if len(X) < 1:
            continue

        if standarized_data:
            #scaler_x.fit(X)
            X = scaler_x.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        raw_predictions = prediction_model(X_tensor)
        r2 = r2_score(y_ground_truth, raw_predictions.detach().numpy())    

        data['Mass Change - predictions'] = raw_predictions.detach().numpy()
        predictions = raw_predictions.detach().numpy()

        predictions_list = [item for sublist in predictions for item in sublist]
        sp.save_results_to_excel(predictions_list, data_file_name, sheet_name, sheet_name, time, y, r2)

"""
Bulk predictions on new data
"""
def bulk_predictions_on_new_data(model_file_name, data_file_name="data/wynikowy_plik_excel.xlsx", standarized_data=False, scaler_x=None, scaler_y=None):
    import re
    from openpyxl import load_workbook
    sp.create_sets_of_new_data_to_predict(data_file_name)   #   This is very important
    
    print(f"Started operations on file {data_file_name}")

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
            #scaler_x.fit(X)
            X = scaler_x.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        raw_predictions = prediction_model(X_tensor)
        
            
        data['Mass Change - predictions'] = raw_predictions.detach().numpy()
        predictions = raw_predictions.detach().numpy()
        predictions_list = [item for sublist in predictions for item in sublist]
        sp.save_results_to_excel_new_data(predictions_list, data_file_name, sheet_name, sheet_name, time)

"""
Load trained model
Temporary function to load graphs to tensor board
"""
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

    dummy_input = torch.rand(1, 10)  
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

### Temporary no needed    
"""
Redundant for now
"""
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