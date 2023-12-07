import data_preparation as dp

BASE_DATA_PATH = "data/data.xlsx"
CLEAR_DATA_PATH = "data/data_cleared.xlsx"
FILTERED_DATA = "data/filtered_data.xlsx"
STANDARIZED_DATA = "data/standarized_data.xlsx"

'''
dp.clear_data(BASE_DATA_PATH, CLEAR_DATA_PATH)
dp.calculate_correlations(CLEAR_DATA_PATH)
dp.create_histograms_data(CLEAR_DATA_PATH)
dp.create_scatters_plot(CLEAR_DATA_PATH)
dp.create_box_diagrams(CLEAR_DATA_PATH)
dp.PCA_analasis(CLEAR_DATA_PATH)
dp.data_analysis(CLEAR_DATA_PATH)
'''
dp.outliers_data_handling(CLEAR_DATA_PATH)
dp.standarize_data(FILTERED_DATA)
