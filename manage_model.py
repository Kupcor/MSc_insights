'''
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

#   Data operations and visualisations libraries

#   Tensor board
from torch.utils.tensorboard import SummaryWriter

#   Custom functions
import data_preparation as sfdp
import hyper_parameters as hp
import snippets as sp
import prepare_outputs as po
import data_features_analisys as dfa


#   !!!! Here is simple instruction to choose model
#   Just change module name, e.g if you want to use model_5_layer architecture use import model_5_layers as model
import model_1_layer as model

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

def run_model_on_test_data():
    X_tensor_2, y_tensor_2 = sfdp.get_test_data_converted_to_tensor(hp.TEST_DATA)
    test_dataset_2 = torch.utils.data.TensorDataset(X_tensor_2, y_tensor_2)
    test_loader_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=hp.batch_size)
    for batch_X, batch_y in test_loader_2:
        print("Batch X:", len(batch_X))
        print("Batch y:", len(batch_y))
    run_loaded_model("Best_models/model_v10_p_0.001_1500_AdamW_testLoss_1.6069554090499878_2023-09-23.pth", "model_v9_p_0.001_3000_Adam_2023-08-30", test_loader=test_loader_2)

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
    material = []
    for prediction in predictions_graph:
        material.append(prediction[0])
    print(material)
    po.create_graph_of_material_change_over_time(time_column, predictions_graph)

    dummy_input = torch.randn(1, input_size).to(hp.device)
    writer.add_graph(prediction_model, dummy_input)
    writer.close()
    #dfa.plot_feature_importance(model)

#run_model()
#cross_validation()
#run_loaded_model("model_v11_model_0.0001_2000_AdamW_2023-09-28_testLoss_0.8954169750213623.pth", test_loader=train_loader)
#run_model_on_test_data()
