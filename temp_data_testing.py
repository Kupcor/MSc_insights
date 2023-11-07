import data_preparation as dp

X, y, x_scaler, y_scaler = dp.get_standarized_data("data/data.xlsx")
print(X)