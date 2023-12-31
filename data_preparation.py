import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib

"""
Simple function for outlier handling
"""
def handle_outliers(data):
    mean = np.mean(data['Mass Change [mg.cm2]'])
    std = np.std(data['Mass Change [mg.cm2]'])
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    data = data[(data['Mass Change [mg.cm2]'] >= lower_bound) & (data['Mass Change [mg.cm2]'] <= upper_bound)]
    return data

"""
Load training data
"""
def load_training_data(file_name, handle_outliers=False):
    data = pd.read_excel(file_name)

    for column in data.columns:
            data[column] = data[column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

    if handle_outliers:
        data = handle_outliers(data)

    data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
    X = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']].values
    y = data['Mass Change [mg.cm2]'].values
    return X, y

"""
Get standarized data
Data standarization with standard scaler from sklearn library
"""
def get_standarized_data(file_name, output_scaling = False):
    X, y = load_training_data(file_name)  

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    if output_scaling:
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

        # Zapisz skalery do plików .pkl
        joblib.dump(x_scaler, 'data/scaler_x.pkl')
        joblib.dump(y_scaler, 'data/scaler_y.pkl')

        return X_scaled, y_scaled, x_scaler, y_scaler

    joblib.dump(x_scaler, 'data/scaler_x.pkl')

    output_excel = "data/current_standarized_data.xlsx"
    data_dict = {'X': X_scaled.tolist(), 'y': y.tolist()}
    df = pd.DataFrame(data_dict)
    df.to_excel(output_excel, index=False)

    return X_scaled, y, x_scaler, None

"""
Get scaler
Get used scallers for standarization 
"""
def get_scaler(output_scaling = False):
    x_scaler = joblib.load('data/scaler_x.pkl')
    
    if output_scaling:
        y_scaled = joblib.load('data/scaler_y.pkl')
        return x_scaler, y_scaled
    
    return x_scaler, None

"""
Get test data converted to tensor
Convert datt to tensors used by pyTorch
"""
def get_test_data_converted_to_tensor(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

def get_splited_training_data(X, y, seed, split_rate):
    X_tensor, y_tensor = get_test_data_converted_to_tensor(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=split_rate, random_state=seed)
    return X_train, X_test, y_train, y_test


"""
For now I will remain those functions
"""
# ________________ Before refactoring
# Sparametryzuj poniższe funkcje, żeby przyjmowały wartości, a nie file name
def get_splitted_training_test_and_validation_data(X, y, seed, split_rate):
    X_tensor, y_tensor = get_test_data_converted_to_tensor(X, y)
    X_train, X_rest, y_train, y_rest = train_test_split(X_tensor, y_tensor, test_size=split_rate, random_state=seed)
    X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)
    return X_train, X_test, X_validation, y_train, y_test, y_validation

def get_only_test_data_as_a_tensor(file_name):
    data = pd.read_excel(file_name)
    X = data[['Temperature [C]', 'Zr [at%]', 'Nb [at%]', 'Mo [at%]', 'Cr [at%]', 'Al [at%]', 'Ti [at%]', 'Ta [at%]', 'W [at%]', 'Time [h]']].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor

def get_time_and_mass_change(file_name):
    data = pd.read_excel(file_name)
    data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
    data.dropna(subset=['Mass Change [mg.cm2]'], inplace=True)
    X = data['Time [h]'].values
    y = data['Mass Change [mg.cm2]'].values
    return X, y


# Maybe helpful in the future
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
