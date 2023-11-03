import pandas as pd
from openpyxl import Workbook

# Wczytaj dane z arkusza Excel
data = pd.read_excel("data/temp.xlsx")

# Stwórz nowy arkusz Excel
wb = Workbook()

# Dla każdej kolumny w danych
for index, row in data.iterrows():
    # Tworzymy nowy arkusz w arkuszu Excel
    ws = wb.create_sheet(title=f"Arkusz_{index}")

    # Dodaj nagłówki kolumn (bez "Czas")
    headers = list(data.columns)
    ws.append(headers[:9] + ["Time [h]"] + headers[9:])

    # Dodaj kolumnę "czas" od 0.1 do 2.3 z krokiem 0.1
    czas = [round(x * 0.1, 1) for x in range(1, 241)]

    # Dodaj wartości z pierwszego wiersza i czas w 10. kolumnie
    for t in czas:
        row_with_time = list(row[:9]) + [t] + list(row[9:])
        ws.append(row_with_time)

# Usuń domyślny arkusz
default_sheet = wb.active
wb.remove(default_sheet)

# Zapisz nowy arkusz Excel
wb.save("data/wynikowy_plik_excel.xlsx")
