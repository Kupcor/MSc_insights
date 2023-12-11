import shap
import torch
import ANN_model_template as model
import hyper_parameters as hp  
import re
import snippets as sp
import data_preparation as dp
import numpy as np
from matplotlib import pyplot as plt

def plot_shap_bar(shap_values, feature_names):
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_indices = np.argsort(mean_shap)[::-1]
    sorted_shap_values = mean_shap[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_shap_values)), sorted_shap_values, color='skyblue')
    plt.yticks(range(len(sorted_shap_values)), sorted_feature_names)
    plt.gca().invert_yaxis()
    plt.xlabel('Średnia wartość bezwzględna SHAP')
    plt.title('Ważność cech według wartości SHAP')
    plt.show()

file_path, model_file_name = sp.select_file()
    
pattern = r'hidden_layers_(.*?)_'
neurons = re.search(pattern, model_file_name).group(1).split("-")
hidden_layers_neuron = [int(neuron) for neuron in neurons]

saved_model_path = f'trained_models/{model_file_name}'
prediction_model = model.PredictionModel(hidden_layers_neuron)
prediction_model.load_state_dict(torch.load(saved_model_path))
prediction_model = prediction_model.to(hp.device)
prediction_model.eval()

X, y, hp.x_scaler, hp.y_scaler = dp.get_standarized_data(hp.DATA_FILE)
#X, y = dp.load_training_data(hp.DATA_FILE)
X_train, X_test, y_train, y_test = dp.get_splited_training_data(X, y, hp.SEED, hp.train_size_rate)
input_size = X_train.shape[1]

explainer = shap.DeepExplainer(prediction_model, X_train)
#explainer = shap.Explainer(prediction_model, X_train)

features_names = ['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]', 'Time [h]']

shap_values = explainer.shap_values(X_train)
shap_values_standard = shap_values.astype(float).tolist()
print(shap_values)
print(f"{len(shap_values)} | {len(shap_values[0])}")
X_train_numpy = X_train.cpu().numpy()

mean_shap_values = np.abs(shap_values).mean(axis=0)
plot_shap_bar(shap_values, features_names)
shap.bar_plot(mean_shap_values)

#shap.text_plot(shap_values)
#shap_plot = shap.force_plot(explainer.expected_value[0], shap_values_standard[0], X_train_numpy[0])
#shap.save_html("shap_plot.html", shap_plot)