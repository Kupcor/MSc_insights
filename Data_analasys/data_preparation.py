import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def clear_data(file_name):
        data = pd.read_excel(file_name)

        for column in data.columns:
                data[column] = data[column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
        data_instance = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]', ['Mass Change [mg.cm2]']]
        data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
        data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']] = pd.to_numeric(data[])

        data_filtered = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]', 'Mass Change [mg.cm2]']].values
        print(data_filtered)
        data_filtered = data_filtered.dropna(subset=['Mass Change [mg.cm2]'])
        
        data.to_excel('data_clear.xlsx', index=False)

def load_data(file_name):
        ata = pd.read_excel(file_name)

        for column in data.columns:
                data[column] = data[column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

        data['Mass Change [mg.cm2]'] = pd.to_numeric(data['Mass Change [mg.cm2]'], errors='coerce')
        data = data.dropna(subset=['Mass Change [mg.cm2]'])
        X = data[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']].values
        y = data['Mass Change [mg.cm2]'].values
        return X, y
