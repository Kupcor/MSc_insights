import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_TYPE = "torch.float64"

def load_training_data(file_name):
    data = pd.read_excel(file_name)

    for column in data.columns:
            data[column] = data[column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

    #   3 sigma
    mean = np.mean(data['Mass Change [mg.cm2]'])
    std = np.std(data['Mass Change [mg.cm2]'])
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    data = data[(data['Mass Change [mg.cm2]'] >= lower_bound) & (data['Mass Change [mg.cm2]'] <= upper_bound)]

    data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
    #data.dropna(subset=['Mass Change [mg.cm2]'], inplace=True)
    X = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']].values
    y = data['Mass Change [mg.cm2]'].values
    return X, y

def get_time_and_mass_change(file_name):
    data = pd.read_excel(file_name)
    data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
    data.dropna(subset=['Mass Change [mg.cm2]'], inplace=True)
    X = data['Time [h]'].values
    y = data['Mass Change [mg.cm2]'].values
    return X, y

def get_test_data_converted_to_tensor(file_name):
    X, y = load_training_data(file_name)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

def get_splited_training_data(file_name, seed, split_rate):
    X_tensor, y_tensor = get_test_data_converted_to_tensor(file_name)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=split_rate, random_state=seed)
    return X_train, X_test, y_train, y_test

def get_splitted_training_test_and_validation_data(file_name, seed, split_rate, data_normalization=False):
    X_tensor, y_tensor = get_test_data_converted_to_tensor(file_name)
    if data_normalization:
        X_tensor, y_tensor = data_normalization_mean(X_tensor=X_tensor, y_tensor=y_tensor)
    X_train, X_rest, y_train, y_rest = train_test_split(X_tensor, y_tensor, test_size=split_rate, random_state=seed)
    X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


def get_only_test_data_as_a_tensor(file_name):
    data = pd.read_excel(file_name)
    X = data[['Temperature [C]', 'Zr [at%]', 'Nb [at%]', 'Mo [at%]', 'Cr [at%]', 'Al [at%]', 'Ti [at%]', 'Ta [at%]', 'W [at%]', 'Time [h]']].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor

def get_test_and_train_loader(file_name, seed, training_rate, batch_size, shuffle):
    X_train, X_test, y_train, y_test = get_splited_training_data(file_name, seed, training_rate)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

def get_test_loader(file_name, batch_size, shuffle):
    X_tensor, y_tensor = get_test_data_converted_to_tensor(file_name)
    test_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


def data_normalization_mean(X_tensor, y_tensor=None):
    mean_X = torch.mean(X_tensor, dim=0)
    std_X = torch.std(X_tensor, dim=0)
    std_X = std_X + 1e-6

    X_standardized = (X_tensor - mean_X) / std_X


    return X_standardized, y_tensor

def input_normalization_mean(X_tensor):
    mean_X = torch.mean(X_tensor, dim=0)
    std_X = torch.std(X_tensor, dim=0)
    std_X = std_X + 1e-6

    X_standardized = (X_tensor - mean_X) / std_X
    return X_standardized