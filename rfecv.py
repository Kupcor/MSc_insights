from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

import hyper_parameters as hp
import data_preparation as dp

import ANN_model_template as model
import ANN_wrapper as aw

X, y, hp.x_scaler, hp.y_scaler = dp.get_standarized_data(hp.DATA_FILE)
#X, y = dp.load_training_data(hp.DATA_FILE)
X_train, X_test, y_train, y_test = dp.get_splited_training_data(X, y, hp.SEED, hp.train_size_rate)
input_size = X_train.shape[1]

def rfecv(X_train = X_train, y_train = y_train, X_test=X_test, y_test=y_test, X_val=None, y_val=None, hidden_layers_neurons = hp.neurons_in_hidden_layers, num_epochs=hp.num_epochs, learning_rate=hp.lr, opt_func=hp.optimizer_arg, model_save=True, show_output_data=True, device=hp.device, input_size=input_size):
    
    prediction_model = model.PredictionModel(hidden_layers_neurons = hidden_layers_neurons, is_dropout=False, input_size=input_size)

    rfecv = RFECV(estimator=prediction_model, cv=5)  

    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.support_]

    X_train_selected = X_train[selected_features]

    print(X_train_selected)

rfecv()