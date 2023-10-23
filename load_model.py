import data_preparation as dp
import hyper_parameters as hp
import torch.nn as nn
import ANN_wrapper as aw


#   Standard split (train and test data) - it is used in base training (without cross validation)
X_train, X_test, X_validation, y_train, y_test, y_validation = dp.get_splitted_training_test_and_validation_data(hp.DATA_FILE, hp.SEED, hp.train_size_rate)


#print(X_test)

aw.load_trained_model()
#aw.hyper_parameter_training()