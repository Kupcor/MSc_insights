import ANN_wrapper as aw
import snippets as sp
import data_preparation as dp
import sys 

aw.load_trained_model()

if len(sys.argv) > 1:
    #aw.cross_validate()
    aw.train_model_wrapper(model_save=True)
    #aw.train_model_with_cross_validation()
    #aw.bulk_training()