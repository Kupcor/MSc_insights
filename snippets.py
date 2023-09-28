import hyper_parameters as hp
import torch.optim as optim

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
    elif opt_arg == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr)
    elif opt_arg == "SGD":
        optimizer = optim.SGD(model.parameters(), lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr)
    return optimizer