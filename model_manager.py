import ANN_wrapper as aw
import snippets as sp
import data_preparation as dp
import sys 



if len(sys.argv) > 1:
    aw.train_model_wrapper(model_save=True)
    #aw.train_model_with_cross_validation()
    #aw.bulk_training()
else:
    file_path, file_name = sp.select_file()
    standaryzacja=True
    scaler_x, scaler_y = dp.get_scaler(output_scaling=True)
    aw.bulk_predictions_on_new_data(file_name, "new_data/test_1_new_data.xlsx", standaryzacja, scaler_x, scaler_y)
    aw.bulk_predictions(file_name, "new_data/test_1_standard_data.xlsx", standaryzacja, scaler_x, scaler_y)
