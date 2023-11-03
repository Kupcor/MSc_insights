import hyper_parameters as hp
import torch.optim as optim
import torch

#   ____________________________    Select Optimizer  _________________________________
#   By defaoult optimizer is Adam
def select_optimizer(model, opt_arg=hp.optimizer_arg, lr=hp.lr):
    if opt_arg == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif opt_arg == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr)
    elif opt_arg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr)
    elif opt_arg == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    # Not aplicable for now
    #elif opt_arg == "LBFGS":
    #    optimizer = optim.LBFGS(model.parameters(), lr, max_iter = 50)
    elif opt_arg == "SGD":
        optimizer = optim.SGD(model.parameters(), lr)
    elif opt_arg == "Nadam":
        optimizer = optim.NAdam(model.parameters(), lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr)
    return optimizer

def select_activation_function(activation_function):
    if activation_function == "ReLU":
        return 

def save_model(model, hidden_layers_neurons, learning_rate, num_epochs, optimizer, test_loss):
    neurons_str = '-'.join(map(str, hidden_layers_neurons))
    file_name = "model_{}_hidden_layers_{}_{}_{}_{}_{}_validation_testLoss_{}".format(
    len(hidden_layers_neurons),
    neurons_str,
    learning_rate,
    num_epochs,
    optimizer.__class__.__name__,
    hp.today,
    test_loss
    )
    save_path = "trained_models/" + file_name + '.pth'
    torch.save(model.state_dict(), save_path)

def save_results_to_excel(predictions, file_name, sheet_name, combination, time, ground_truth):
    from openpyxl import load_workbook
    import openpyxl
    import matplotlib.pyplot as plt

    workbook = load_workbook(file_name)

    if sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
    else:
        sheet = workbook.create_sheet(title=sheet_name)

    sheet.cell(row=1, column=12, value="Mass change - predictions")

    for i, prediction in enumerate(predictions, start=2):
        sheet.cell(row=i, column=12, value=prediction)

    fig, ax = plt.subplots()
    ax.scatter(time, predictions, label='Predictions')
    ax.scatter(time, ground_truth, label='Ground Truth')

    # Dodaj etykiety i legendę
    ax.set_xlabel('Time')
    ax.set_ylabel('Mass Change')
    ax.set_title(combination)
    ax.legend()
    ax.grid()

    # Zapisz rysunek do arkusza
    img_path = f'image/{sheet_name}.png' 
    plt.savefig(img_path)

    img = openpyxl.drawing.image.Image(img_path)
    sheet.add_image(img, 'M10') 
    workbook.save(file_name)

def save_results_to_excel_new_data(predictions, file_name, sheet_name, combination, time):
    from openpyxl import load_workbook
    import openpyxl
    import matplotlib.pyplot as plt

    workbook = load_workbook(file_name)

    if sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
    else:
        sheet = workbook.create_sheet(title=sheet_name)

    sheet.cell(row=1, column=12, value="Mass change - predictions")

    for i, prediction in enumerate(predictions, start=2):
        sheet.cell(row=i, column=12, value=prediction)

    fig, ax = plt.subplots()
    ax.scatter(time, predictions, label='Predictions', s=10)

    # Dodaj etykiety i legendę
    ax.set_xlabel('Time')
    ax.set_ylabel('Mass Change')
    ax.set_title(combination)
    ax.legend()
    ax.grid()

    # Zapisz rysunek do arkusza
    img_path = f'image/{sheet_name}.png' 
    plt.savefig(img_path)

    img = openpyxl.drawing.image.Image(img_path)
    sheet.add_image(img, 'M10') 
    workbook.save(file_name)

## !!! Important function
def data_reader(file_name='bulk_results'):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_excel('data/data.xlsx')

    wybrane_kolumny = df[['Temperature [C]', 'Mo [at%]', 'Nb [at%]', 'Ta [at%]', 'Ti [at%]', 'Cr [at%]', 'Al [at%]', 'W [at%]', 'Zr [at%]']]

    unikalne_kombinacje = wybrane_kolumny.drop_duplicates()
    with pd.ExcelWriter(f'{file_name}', engine='xlsxwriter') as writer:
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
                'marker': {'type': 'circle', 'size': 2},
            })
            chart.set_title({'name': combination})
            chart.set_x_axis({'name': 'Time'})
            chart.set_y_axis({'name': 'Mass Change'})
            worksheet.insert_chart('N2', chart)

def redundant_func(file_name):
    import pandas as pd
    from openpyxl import Workbook

    data = pd.read_excel("data/temp.xlsx")

    wb = Workbook()

    for index, row in data.iterrows():
        ws = wb.create_sheet(title=f"Arkusz_{index}")

        headers = list(data.columns)
        ws.append(headers[:9] + ["Time [h]"] + headers[9:])

        czas = [round(x * 0.1, 1) for x in range(0, 241)]

        for t in czas:
            row_with_time = list(row[:9]) + [t] + list(row[9:])
            ws.append(row_with_time)

    default_sheet = wb.active
    wb.remove(default_sheet)

    wb.save(f"{file_name}")

def select_file():
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir=r"C:\Users\Piotr Kupczyk\Mój folder\Studia\Informatyczna techniczna\Praca magisterska\Model\trained_models")
    if file_path:
        file_name = file_path.split("/")[-1]
    else:
        return None
    
    print(f"{file_path}")
    return file_path, file_name
