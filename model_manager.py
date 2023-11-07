import ANN_wrapper as aw
import snippets as sp
import data_preparation as dp
import sys 

standaryzacja = True


if len(sys.argv) > 1:
    aw.train_model_wrapper(model_save=True)
    #aw.train_model_with_cross_validation()
else:
    file_path, file_name = sp.select_file()
    scaler_x, scaler_y = dp.get_scaler("data/data.xlsx")
    aw.bulk_predictions_on_new_data(file_name, "new_data/test_1_new_data.xlsx", standaryzacja, scaler_x)
    aw.bulk_predictions(file_name, "new_data/test_1_standard_data.xlsx", standaryzacja, scaler_x)
