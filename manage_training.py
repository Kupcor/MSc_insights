import manage_model as m
import hyper_parameters as hp
import data_preparation as dp
import sys


current_model = "main_model_1_layers_model_0.001_1500_Adam_2023-10-15_testLoss_673.5739135742188"
current_model = current_model + ".pth"

#m.load_model("model_v11_model_0.0001_2000_AdamW_2023-09-28_testLoss_0.8954169750213623.pth", hp.TESTING_FILE)
#print(dp.get_only_test_data_as_a_tensor(hp.TESTING_FILE))



if len(sys.argv) > 1:
    m.load_model(current_model, hp.TESTING_FILE)
else:
    m.run_model()


