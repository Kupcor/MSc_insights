import ANN_wrapper as aw
import snippets as sp
import sys 


if len(sys.argv) > 1:
    aw.train_model_wrapper(model_save=True)
    #aw.train_model_with_cross_validation()
else:
    file_path, file_name = sp.select_file()
    aw.bulk_predictions_on_new_data(file_path, file_name, "new_data/test_1_new_data.xlsx")
    aw.bulk_predictions(file_path, file_name, "new_data/test_1_standard_data.xlsx")
