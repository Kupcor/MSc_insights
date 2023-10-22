import manage_model as m
import hyper_parameters as hp
import data_preparation as dp
import sys


current_model = "main_model_1_layers_model_0.001_20000_AdamW_2023-10-21_testLoss_13.814801216125488_one_layer_neuron_num_12"
current_model = current_model + ".pth"

cc= "model_v11_model_0.0001_2000_AdamW_2023-09-28_testLoss_0.8954169750213623"
cc = cc + ".pth"

#m.load_model("model_v11_model_0.0001_2000_AdamW_2023-09-28_testLoss_0.8954169750213623.pth", hp.TESTING_FILE)
#print(dp.get_only_test_data_as_a_tensor(hp.TESTING_FILE))
neuron_number = 12


if len(sys.argv) == 1:
    m.load_model_one_layer(current_model, hp.TESTING_FILE, neuron_number)
    #m.load_model(cc, hp.TESTING_FILE)
else:
    #m.run_model()
    #m.train_multiple_models()
    m.train_parametrized_one_layer_model(neuron_number)
    #m.train_hyperparameters()
