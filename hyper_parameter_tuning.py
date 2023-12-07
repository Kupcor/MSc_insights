import random

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
import torch.optim as optim

def generate_hyperparameter_combinations(num_combinations):
    hyperparameter_sets = []

    for _ in range(num_combinations):
        hidden_layers_neurons = []
        num_hidden_layers = random.randint(1, 3)

        for _ in range(num_hidden_layers):
            num_neurons = random.randint(8, 32)
            hidden_layers_neurons.append(num_neurons)

        num_epochs = random.randint(2000, 6000)
        learning_rate = round(random.uniform(0.0001, 0.01), 3)
        optimizer_choices = ['Adam', 'SGD', 'RMSprop', "AdamW"]
        opt_func = random.choice(optimizer_choices)

        hyperparameter_set = {
            'hidden_layers_neurons': hidden_layers_neurons,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'opt_func': opt_func
        }

        hyperparameter_sets.append(hyperparameter_set)

    return hyperparameter_sets
