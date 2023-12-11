import ANN_wrapper as aw
import snippets as sp
import data_preparation as dp
import sys 


if len(sys.argv) > 1:
    #aw.cross_validate()
    #aw.load_trained_model()
    aw.train_model_wrapper(model_save=True)
    #aw.train_model_with_cross_validation()
    #aw.bulk_training()
else:
    file_path, file_name = sp.select_file()
    print(file_name)
    standaryzacja=True
    scaler_x, scaler_y = dp.get_scaler(output_scaling=False)
    #aw.bulk_predictions_on_new_data(file_name, "new_data/test_1_new_data.xlsx", standaryzacja, scaler_x, scaler_y)
    aw.bulk_predictions(file_name, "new_data/test_1_standard_data.xlsx", standaryzacja, scaler_x, scaler_y)
"""

import snippets as sp
import data_preparation as dp
import sys
import torch

import tkinter as tk

def train_model():
    import ANN_wrapper as aw
    import hyper_parameters as hp
    param_string = entry.get()
    lr, num_epochs, optimizer_arg, seed, train_size_rate, neurons_in_hidden_layers, activation_func_1 = hp.set_hyperparameters_from_string(param_string)
    aw.train_model_wrapper(hidden_layers_neurons = neurons_in_hidden_layers, num_epochs=num_epochs, learning_rate=lr, opt_func=optimizer_arg, model_save=True, show_output_data=True, device=hp.device, activation_func=activation_func_1)

def prepare_data():
    import ANN_wrapper as aw
    file_path, file_name = sp.select_file()
    standaryzacja=True
    scaler_x, scaler_y = dp.get_scaler(output_scaling=False)
    aw.bulk_predictions_on_new_data(file_name, "new_data/test_1_new_data.xlsx", standaryzacja, scaler_x, scaler_y)
    aw.bulk_predictions(file_name, "new_data/test_1_standard_data.xlsx", standaryzacja, scaler_x, scaler_y)
    label.config(text="Prepare data")

def load_model():
    label.config(text="load_model")
    import ANN_wrapper as aw
    aw.load_trained_model()

root = tk.Tk()
root.title("Quick gui")

label = tk.Label(root, text="")
label.pack()

button1 = tk.Button(root, text="Train model", command=train_model)
entry = tk.Entry(root, width=50)
entry.pack()
button2 = tk.Button(root, text="Prepare data", command=prepare_data)
button3 = tk.Button(root, text="Load data", command=load_model)

button1.pack()
button2.pack()
button3.pack()

root.mainloop()
"""
