import hyper_parameters as hp
import torch.optim as optim
import torch

#   ____________________________    Select Optimizer  _________________________________
#   By defaoult optimizer is Adam
def select_optimizer(model, opt_arg=hp.optimizer_arg, lr=hp.lr):
    if opt_arg == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif opt_arg == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr)
    elif opt_arg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr)
    elif opt_arg == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    # Not aplicable for now
    #elif opt_arg == "LBFGS":
    #    optimizer = optim.LBFGS(model.parameters(), lr, max_iter = 50)
    elif opt_arg == "SGD":
        optimizer = optim.SGD(model.parameters(), lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr)
    return optimizer

def select_activation_function(activation_function):
    if activation_function == "ReLU":
        return 

def save_model(model, hidden_layers_neurons, learning_rate, num_epochs, optimizer, test_loss, accuracy, average_loss):
    neurons_str = '-'.join(map(str, hidden_layers_neurons))
    file_name = "model_{}_hidden_layers_{}_{}_{}_{}_{}_validation_accuracy_{:.2f}_avloss_{:.2f}_testLoss_{:.2f}".format(
    len(hidden_layers_neurons),
    neurons_str,
    learning_rate,
    num_epochs,
    optimizer.__class__.__name__,
    hp.today,
    accuracy,
    average_loss,
    test_loss
    )
    save_path = "trained_models/" + file_name + '.pth'
    torch.save(model.state_dict(), save_path)
