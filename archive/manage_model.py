#   Standard python libraries
from datetime import date

#   PyTorch modules
import torch
import torch.nn as nn

#   Data operations and visualisations libraries

#   Tensor board
from torch.utils.tensorboard import SummaryWriter

#   Custom functions
import data_preparation as sfdp
import hyper_parameters as hp
import snippets as sp
import prepare_outputs as po

#   Here is simple instruction to choose model
#   Just change module name, e.g if you want to use model_5_layer architecture use import model_5_layers as model
import model_1_layer_parametrized as model
#import model_5_layers as model


#   ____________________________    Load Data  _________________________________
#   Standard split (train and test data) - it is used in base training (without cross validation)
X_train, X_test, y_train, y_test = sfdp.get_splited_training_data(hp.DATA_FILE, hp.SEED, hp.train_size_rate)
#   Create data sets (in data loaders form)
train_loader, test_loader = sfdp.get_test_and_train_loader(hp.DATA_FILE, hp.SEED, hp.train_size_rate, hp.batch_size, True)
#   Input size
for batch_X, batch_y in test_loader:
    input_size = batch_X.shape[1]

#   Standard run
def run_model(train_loader = train_loader, test_loader = test_loader, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg):
    prediction_model = model.PredictionModel(input_size)
    prediction_model = prediction_model.to(hp.device)
    loss_function = nn.MSELoss() 
    
    #   Select optimizer
    optimizer = sp.select_optimizer(prediction_model, opt_arg=opt_func, lr=learning_rate)
    #   Traing model
    train_loss = model.train_model(prediction_model, train_loader, test_loader, loss_function, optimizer, num_epochs)
    #   Test model
    test_predictions, real_results, test_loss, input_data, losses = model.test_model(prediction_model, test_loader, loss_function)

    #   Show plots
    po.plot_predictions(test_data=real_results, predictions=test_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
    po.loss_oscilation(train_loss, opt=opt_func, epochs=num_epochs, lr=learning_rate)

    #   Save model
    file_name = f"_model_{learning_rate}_{num_epochs}_{optimizer.__class__.__name__}_{hp.today}_testLoss_{test_loss}"
    save_path = "models/" + hp.MODEL_NAME + "_" + prediction_model.model_name + file_name + '.pth'
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
    model = model.PredictionModel(input_size)
    model.load_state_dict(torch.load(saved_model_path))
    model = model.to(hp.device)

    #dfa.plot_feature_importance(model, X=X_test, feature_names = hp.FEATURES)

    model.eval()
    predictions, real_values, test_loss, input_data, losses = model.test_model(test_loader=test_loader, model=model, loss_fun=nn.MSELoss(), show_results=True)
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

def load_model(file_name, data_file, test_loader=test_loader, loss_fun = nn.MSELoss(), X_val = X_test):
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
    prediction_model = model.PredictionModel(input_size)
    prediction_model.load_state_dict(torch.load(saved_model_path))
    prediction_model = prediction_model.to(hp.device)

    prediction_model.eval()
    X_data = sfdp.get_only_test_data_as_a_tensor(data_file)
    X_tensor = X_data.to(hp.device)
    predictions = prediction_model(X_tensor)

    time_column = X_tensor.cpu().numpy()
    time_column = time_column[:, -1]
    
    predictions_graph = predictions.cpu().detach().numpy()

    X_reference_data, y_reference_data = sfdp.get_time_and_mass_change("data/rdata.xlsx")
    time_column_ref = X_reference_data
    predictions_graph_ref = y_reference_data
    
    po.create_graph_of_material_change_over_time(time_column, predictions_graph, time_column_ref, predictions_graph_ref)

    dummy_input = torch.randn(1, input_size).to(hp.device)
    writer.add_graph(prediction_model, dummy_input)
    writer.close()
    #dfa.plot_feature_importance(model)

def train_multiple_models(train_loader = train_loader, test_loader = test_loader, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg):
    values = {}
    #   q = m - 1 / n + 2 => m - number of train samples, n - input features
    for neuron_number in range(7, 15):
        print(f'________\nTraining model with {neuron_number} neurons start')
        prediction_model = model.PredictionModel(input_size, neuron_number)
        prediction_model = prediction_model.to(hp.device)
        loss_function = nn.MSELoss()

        #   Select optimizer
        optimizer = sp.select_optimizer(prediction_model, opt_arg=opt_func, lr=learning_rate)
        #   Traing model
        train_loss = model.train_model(prediction_model, train_loader, test_loader, loss_function, optimizer, num_epochs)
        #   Test model
        test_predictions, real_results, test_loss, input_data, losses = model.test_model(prediction_model, test_loader, loss_function)

        key = f'layer_number | {neuron_number}'
        value = f'loss: {test_loss}'
        values[key] = value

    
    with open(f'hyperparameters_searching/test_data_{opt_func}_{learning_rate}_{num_epochs}_activation_tahn_wyjscia_wykl.txt', "w") as file:
        file.write(f'Results training results:\n')
        for key in values.keys():
            formatted_line = f'{key} : {values[key]}\n'
            file.write(formatted_line)

    # Show plots
    # po.plot_predictions(test_data=real_results, predictions=test_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
    # po.loss_oscilation(train_loss, opt=opt_func, epochs=num_epochs, lr=learning_rate)

    # Save model
    # file_name = f"_model_{learning_rate}_{num_epochs}_{optimizer.__class__.__name__}_{hp.today}_testLoss_{test_loss}"
    # save_path = "models/" + hp.MODEL_NAME + "_" + prediction_model.model_name + file_name + '.pth'
    # torch.save(prediction_model.state_dict(), save_path)

# Search for hyperparameters and other functions
def train_parametrized_one_layer_model(layer_number, train_loader = train_loader, test_loader = test_loader, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg):
    neuron_number = layer_number
    #   q = m - 1 / n + 2 => m - number of train samples, n - input features
    print(f'________\nTraining model with {neuron_number} neurons start')
    prediction_model = model.PredictionModel(input_size, neuron_number)
    prediction_model = prediction_model.to(hp.device)
    loss_function = nn.MSELoss()

    #   Select optimizer
    optimizer = sp.select_optimizer(prediction_model, opt_arg=opt_func, lr=learning_rate)
    #   Traing model
    train_loss = model.train_model(prediction_model, train_loader, test_loader, loss_function, optimizer, num_epochs, X_tensor = X_train, y_tensor = y_train)
    #   Test model
    test_predictions, real_results, test_loss, input_data, losses = model.test_model(prediction_model, test_loader, X_test, y_test, loss_function)

    #po.plot_predictions(test_data=real_results, predictions=test_predictions, opt=opt_func, lr=learning_rate, epochs=num_epochs)
    po.loss_oscilation(train_loss, opt=opt_func, epochs=num_epochs, lr=learning_rate)

    file_name = f"_model_{learning_rate}_{num_epochs}_{optimizer.__class__.__name__}_{hp.today}_testLoss_{test_loss}_one_layer"
    save_path = f'one_layer_models/{hp.MODEL_NAME}_{prediction_model.model_name}{file_name}_neuron_num_{neuron_number}.pth'
    torch.save(prediction_model.state_dict(), save_path)
    return train_loss

def train_hyperparameters():
    values = {}
    lrs = [0.01, 0.001, 0.0001, 0.00001]
    epochs = [500, 1000, 2000, 3000, 4000, 5000]
    optimizers = ["Adam", "AdamW", "RMSprop", "SGD", "Adagrad"]
    for lr in lrs:
        for epoch in epochs:
            for optimizer in optimizers:
                print(f'Start for {lr} {epoch} {optimizer}')
                test_loss = train_parametrized_one_layer_model(layer_number=13, learning_rate=lr, opt_func=optimizer, num_epochs=epoch)
                key = f'{lr}_{epoch}_{optimizer}'
                value = f'loss: {test_loss}'
                values[key] = value

    with open(f'hyperparameters_searching/test_data_hyperparameters_relu_and_batch_normalizations.txt', "w") as file:
        file.write(f'Results training results:\n')
        for key in values.keys():
            formatted_line = f'{key} : {values[key]}\n'
            file.write(formatted_line)

def load_model_one_layer(file_name, data_file, neuron_number, X_tensor = X_train, Y_tensor = y_train, test_loader=test_loader, loss_fun = nn.MSELoss(), X_val = X_test):
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

    saved_model_path = f'one_layer_models/{file_name}'
    prediction_model = model.PredictionModel(input_size, neuron_number)
    prediction_model.load_state_dict(torch.load(saved_model_path))
    prediction_model = prediction_model.to(hp.device)

    prediction_model.eval()

    X_data = sfdp.get_only_test_data_as_a_tensor(data_file)
    X_tensor = X_data.to(hp.device)
    predictions = prediction_model(X_tensor)

    time_column = X_tensor.cpu().numpy()
    time_column = time_column[:, -1]
    
    predictions_graph = predictions.cpu().detach().numpy()

    X_reference_data, y_reference_data = sfdp.get_time_and_mass_change("data/rdata.xlsx")
    time_column_ref = X_reference_data
    predictions_graph_ref = y_reference_data
    
    po.create_graph_of_material_change_over_time(time_column, predictions_graph, time_column_ref, predictions_graph_ref)

    dummy_input = torch.randn(1, input_size).to(hp.device)
    writer.add_graph(prediction_model, dummy_input)
    writer.close()
    #dfa.plot_feature_importance(model)
