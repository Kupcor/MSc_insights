import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_TYPE = "torch.float64"

def load_training_data(file_name):
    data = pd.read_excel(file_name)
    data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
    data.dropna(subset=['Mass Change [mg.cm2]'], inplace=True)
    X = data[['Temperature [C]', 'Zr [at%]', 'Nb [at%]', 'Mo [at%]', 'Cr [at%]', 'Al [at%]', 'Ti [at%]', 'Ta [at%]', 'W [at%]', 'Time [h]']].values
    y = data['Mass Change [mg.cm2]'].values
    return X, y

def get_test_data_converted_to_tensor(file_name):
    X, y = load_training_data(file_name)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

def get_splited_training_data(file_name, seed, training_rate):
    #   Load training data in Tensor formats
    X_tensor, y_tensor = get_test_data_converted_to_tensor(file_name)
    #   Standard split (train and test data) - it is used in base training (without cross validation)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=training_rate,
                                        random_state=seed) #  Set fixed random_state to get "repeated" data in each program run
    return X_train, X_test, y_train, y_test

def get_test_and_train_loader(file_name, seed, training_rate, batch_size, shuffle):
    X_train, X_test, y_train, y_test = get_splited_training_data(file_name, seed, training_rate)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

#   _______________________________________________________________________
#   Below section is redundat for now
def get_normalized_data(file_name):
    X, y =load_training_data(file_name)
    X = Z_score_normalization(X,y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

'''
    Min max normalization
    value - min / max - min
    Pros -> easy and simple
    Coins -> not handle outlieres very well
'''
def Min_Max_normalization(X, y):
    temperature = X[:, 0]
    times = X[:, -1]
    percentages = X[:, 1:-1]

    normalized_temperature = (temperature - temperature.min()) / (temperature.max() - temperature.min())
    normalized_times = (times - times.min()) / (times.max() - times.min())

    #   Percentages normalization by row
    #normalized_percentages = (percentages - percentages.min(axis=1, keepdims=True)) / (percentages.max(axis=1, keepdims=True) - percentages.min(axis=1, keepdims=True))    
    #   Percentages normalization by column
    normalized_percentages = (percentages - percentages.min(axis=0, keepdims=True)) / (percentages.max(axis=0, keepdims=True) - percentages.min(axis=0, keepdims=True))
    
    normalized_data = np.hstack((normalized_temperature.reshape(-1, 1), normalized_percentages, normalized_times.reshape(-1, 1)))
    return normalized_data

'''
    Z score normalization
    value - u / standard_deviation
    Potentail downside: features are not exact on the same scale
'''
def Z_score_normalization(X,y):
    temperature = X[:, 0]
    times = X[:, -1]
    percentages = X[:, 1:-1]

    mean_temperature = np.mean(temperature)
    std_dev_temperature = np.std(temperature)
    normalized_temperature = (temperature - mean_temperature) / std_dev_temperature

    mean_times = np.mean(times)
    std_dev_times = np.std(times)
    normalized_times = (times - mean_times) / std_dev_times

    #   Percentages normalization by row
    normalized_percentages = (percentages - np.mean(percentages, axis=1, keepdims=True)) / np.std(percentages, axis=1, keepdims=True)
    #   Percentages normalization by column
    #   normalized_percentages = (percentages - np.mean(percentages, axis=0, keepdims=True)) / np.std(percentages, axis=0, keepdims=True)
    
    normalized_data = np.hstack((normalized_temperature.reshape(-1, 1), normalized_percentages, normalized_times.reshape(-1, 1)))
    return normalized_data

"""
    Interesting sources:
    - https://www.codecademy.com/article/normalization
"""