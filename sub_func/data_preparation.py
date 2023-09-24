import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

DATA_TYPE = "torch.float64"

def load_data(file_name):
    #   Wczytanie danych do struktury z biblioteki pandas
    data = pd.read_excel(file_name)
    #   Czyszczenie danych
    #   Usunięcie wierszy zawierających puste wartości w zmiennej zależnej "Mass Change"
    data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
    data.dropna(subset=['Mass Change [mg.cm2]'], inplace=True)
    #   Przypisanie danych do odpowiednich podzbiorów
    #   __  Mass Change __ jest zmienną zależną
    X = data[['Temperature [C]', 'Zr [at%]', 'Nb [at%]', 'Mo [at%]', 'Cr [at%]', 'Al [at%]', 'Ti [at%]', 'Ta [at%]', 'W [at%]', 'Time [h]']].values
    y = data['Mass Change [mg.cm2]'].values
    return X, y

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data

def get_prepared_data(file_name):
    X, y = load_data(file_name)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

def get_normalized_data(file_name):
    X, y =load_data(file_name)
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

""" TODO - does it make any sens?
def robust_normalization(X, y):
    temperature = X[:, 0]
    times = X[:, -1]
    percentages = X[:, 1:-1]

    return normalized_data

def L2_normalization(X, y):
    temperature = X[:, 0]
    times = X[:, -1]
    percentages = X[:, 1:-1]

    return normalized_data

def L1_normalization(X, y):
    temperature = X[:, 0]
    times = X[:, -1]
    percentages = X[:, 1:-1]

    return normalized_data
"""


"""
    Interesting sources:
    - https://www.codecademy.com/article/normalization
"""