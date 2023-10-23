import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('data/data.xlsx')

wybrane_kolumny = df[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]']]

unikalne_kombinacje = wybrane_kolumny.drop_duplicates()

with pd.ExcelWriter('data/results_file.xlsx', engine='xlsxwriter') as writer:
    iterator = 1
    for i, row in unikalne_kombinacje.iterrows():
        sheet_name = f"combination_{iterator}"

        combination = ', '.join(row.iloc[:-1].astype(str))

        subset = df[(df['Temperature [C]'] == row['Temperature [C]']) & 
                    (df['Mo [at%]'] == row['Mo [at%]']) & 
                    (df['Nb [at%]'] == row['Nb [at%]']) & 
                    (df['Ta [at%]'] == row['Ta [at%]']) & 
                    (df['Ti [at%]'] == row['Ti [at%]']) & 
                    (df['Cr [at%]'] == row['Cr [at%]']) & 
                    (df['Al [at%]'] == row['Al [at%]']) & 
                    (df['W [at%]'] == row['W [at%]']) & 
                    (df['Zr [at%]'] == row['Zr [at%]'])]

        subset = subset.drop(columns=['Parametr X'])

        subset.to_excel(writer, sheet_name=sheet_name, index=False)
        iterator += 1

        worksheet = writer.sheets[sheet_name]
        chart = writer.book.add_chart({'type': 'scatter'})
        chart.add_series({
            'name': "Ground truth",
            'categories': f"='{sheet_name}'!$J$2:$J${len(subset) + 1}",
            'values': f"='{sheet_name}'!$K$2:$K${len(subset) + 1}",
            'marker': {'type': 'circle', 'size': 5},
        })
        chart.set_title({'name': combination})
        chart.set_x_axis({'name': 'Time'})
        chart.set_y_axis({'name': 'Mass Change'})
        worksheet.insert_chart('N2', chart)