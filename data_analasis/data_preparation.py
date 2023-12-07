import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import math
from sklearn.decomposition import PCA
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

def convert_to_float(x):
    if isinstance(x, str):
        x = x.replace(',', '.')
        try:
            return float(x)
        except ValueError:
            return None
    else:
        return x

def remove_outliers_iqr(df, threshold=1.5):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = df[(df >= lower_bound) & (df <= upper_bound)]
    
    return filtered_df
    
"""
    Clearing data -> removing rows with missing values, converting all data to floats and numeric type, removing redundant columns
"""
def clear_data(file_name, new_file_path):
    data = pd.read_excel(file_name)
    # Drop redundant column
    data.drop('Unnamed: 11', axis=1, inplace=True)
    # Convert data to floats
    for column in data.columns:
            data[column] = data[column].apply(convert_to_float)
    # Removed rows with missing values
    data.dropna(axis='index', how='any', inplace=True)
    # Save cleared data to new file
    data.to_excel(new_file_path, index=False)

"""
Correlations
"""
def calculate_correlations(file_name):
    save_results_path_pdf = "results/correlations.pdf" 
    save_results_path_png = "results/correlations.png" 
    data = pd.read_excel(file_name)
    correlations = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
    plt.yticks(rotation=0)
    plt.subplots_adjust(bottom=0.25)
    plt.title("Correlation heatmap")
    plt.savefig(save_results_path_pdf, format="pdf", dpi=100)
    plt.savefig(save_results_path_png, format="png", dpi=100)
    plt.show()
    print("Correlation Matrix:")
    print(correlations)

"""
    Histograms
"""
def create_histograms_data(file_name):
    save_results_path_pdf = "results/histograms.pdf" 
    save_results_path_png = "results/histograms.png" 

    data = pd.read_excel(file_name)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    axes = axes.flatten()

    bins = int(1 + math.log2(len(data["Time [h]"])))

    for i, column in enumerate(data.columns):
        ax = axes[i]
        ax.hist(data[column], bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        mean, std = norm.fit(data[column])     #   From norm - calculate mean and std
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)    #   Create an array of values evenly spaced between minimum and maximum
        p = norm.pdf(x, mean, std)          #   Calculate y values for normal distribution
        ax.plot(x, p, 'r', linewidth=2, label=f'N({mean:.2f}, {std**2:.2f})')   #   And just plot

    if len(data.columns) < len(axes):
        for i in range(len(data.columns), len(axes)):
            fig.delaxes(axes[i])

    plt.savefig(save_results_path_pdf, format="pdf", dpi=300)
    plt.savefig(save_results_path_png, format="png", dpi=300)
    plt.tight_layout(pad=3.0)
    plt.show()

"""
Box diagrams
"""
def create_box_diagrams(file_name):
    save_results_path_pdf = "results/boxes.pdf" 
    save_results_path_png = "results/boxes.png"
    data = pd.read_excel(file_name)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, column in enumerate(data.columns):
        ax = axes[i]
        colors = sns.color_palette('Set3')
        sns.boxplot(data[column], ax=ax, color=colors[i % len(colors)], showfliers=True)

        ax.set_title(f'Boxplot of {column}')
        ax.set_xlabel(column)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        stats = data[column].describe()
        textstr = (f'Min: {stats["min"]:.2f}\n'
                    f'Q1: {stats["25%"]:.2f}\n'
                    f'Median: {stats["50%"]:.2f}\n'
                    f'Q3: {stats["75%"]:.2f}\n'
                    f'Max: {stats["max"]:.2f}')
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))


    if len(data.columns) < len(axes):
        for i in range(len(data.columns), len(axes)):
            fig.delaxes(axes[i])

    plt.savefig(save_results_path_pdf, format="pdf", dpi=300)
    plt.savefig(save_results_path_png, format="png", dpi=300)
    plt.tight_layout(pad=3.0)
    plt.show()

"""
Scatter plots
"""
def create_scatters_plot(file_name):
    save_results_path_pdf = "results/scatters.pdf" 
    save_results_path_png = "results/scatters.png"
    data = pd.read_excel(file_name)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 12))
    axes = axes.flatten()

    mass_change = data["Mass Change [mg.cm2]"]
    data.drop("Mass Change [mg.cm2]", axis=1, inplace=True)

    for i, column in enumerate(data.columns):
        ax = axes[i]
        ax.scatter(mass_change, data[column], s=2, marker='o', alpha=0.7)

        z = np.polyfit(mass_change, data[column], 1)
        trend_line = np.poly1d(z)
        ax.plot(mass_change, trend_line(mass_change), "r--")

        ax.set_title(f'Scatter plot')
        ax.set_xlabel(column)
        ax.set_ylabel('Mass Change [mg.cm2]')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if len(data.columns) < len(axes):
        for i in range(len(data.columns), len(axes)):
            fig.delaxes(axes[i])

    plt.savefig(save_results_path_pdf, format="pdf", dpi=300)
    plt.savefig(save_results_path_png, format="png", dpi=300)
    plt.tight_layout(pad=2.0)
    plt.show()

"""
Basic statistics
"""
def data_analysis(file_name):
    data = pd.read_excel(file_name)
    stats = pd.DataFrame()
    stats['Non-Null Count'] = data.count()
    stats['% Non-Null'] = (data.count() / len(data)) * 100

    stats['Mean'] = data.mean()
    stats['Median'] = data.median()
    stats['Mode'] = data.mode().iloc[0]
    stats['Count'] = data.count()
    stats['Min'] = data.min()
    stats['Max'] = data.max()
    stats['Variance'] = data.var()
    stats['Std Deviation'] = data.std()
    stats['Std Error Mean'] = data.sem()
    stats['Kurtosis'] = data.kurtosis()

    file_path = 'data/statistic.xlsx'
    stats.to_excel(file_path)

"""
Data standarization
"""
def standarize_data(file_name):
    file_path_to_new_file = "data/standarized_data.xlsx"
    data = pd.read_excel(file_name)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    scaled_df.to_excel(file_path_to_new_file, index=False)

"""
Outliers data handling
"""
def outliers_data_handling(file_name, m=3):
    new_name = "data/filtered_data.xlsx"
    data = pd.read_excel(file_name)
    data = remove_outliers_iqr(data, threshold=m)
    data.dropna(axis='index', how='any', inplace=True)
    data.to_excel(new_name, columns=data.columns, index=False)

#   TODO
def PCA_analasis(file_name):
    save_results_path_pdf = "results/PCA_cumulativesum.pdf" 
    save_results_path_png = "results/PCA_cumulativesum.png"
    data = pd.read_excel(file_name)
    data.drop('Mass Change [mg.cm2]', axis=1, inplace=True)

    data = data.dropna() #  Just to be sure ok?

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)    #   Standarization, mean=0, var = 1

    pca = PCA()
    pca.fit(scaled_data)
    explained_variance = pca.explained_variance_    
    explained_variance_ratio = pca.explained_variance_ratio_
    principal_components = pca.components_

    pca = PCA(n_components=7)
    principal_components = pca.fit_transform(scaled_data)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-', color='b')
    plt.xlabel('Liczba Komponentów Głównych')
    plt.ylabel('Kumulatywna Suma Wyjaśnionej Wariancji')
    plt.title('Wykres Kumulatywnej Sumy Wyjaśnionej Wariancji')
    plt.grid(True)
    plt.savefig(save_results_path_pdf, format="pdf", dpi=100)
    plt.savefig(save_results_path_png, format="png", dpi=100)
    plt.show()

    save_results_path_pdf = "results/PCA_ważności.pdf" 
    save_results_path_png = "results/PCA_ważności.png"

    plt.bar(range(len(pca.components_[0])), pca.components_[0])
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.xlabel('Cechy')
    plt.ylabel('Współczynniki komponentu głównego 1')
    plt.title('Wykres Ważności Cech dla Komponentu Głównego 1')
    plt.grid(True)
    plt.savefig(save_results_path_pdf, format="pdf", dpi=100)
    plt.savefig(save_results_path_png, format="png", dpi=100)
    plt.show()

    data_with_pca = pd.DataFrame(principal_components, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"])
    data_with_pca.to_excel("data/PCA_results_7.xlsx", index=False)


