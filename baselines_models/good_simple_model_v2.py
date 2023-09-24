import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import sys

import sub_func.data_preparation


#   Global parameter -> I know, bad practices bla bla bla
train_size_rate = 0.9
lr = 0.001              #   Learning rate
num_epochs = 1000       #   Definicja ilości epok uczenia

if len(sys.argv) > 1:
    print(f"Lr set to {sys.argv[1]}")
    lr = float(sys.argv[1]) 

    
if len(sys.argv) > 2:
    print(f"Epoch number set to {sys.argv[2]}")
    num_epochs = int(sys.argv[2])
  
#   Load data from data.xlsx file
X_tensor, y_tensor = sub_func.data_preparation.get_prepared_data("data.xlsx")

#   Step 6 | 7
#   Split data to train and test data
train_size = int(train_size_rate * len(X_tensor))
test_size = len(X_tensor) - train_size

X_train, X_test = torch.split(X_tensor, [train_size, test_size])
y_train, y_test = torch.split(y_tensor, [train_size, test_size])

#   Step 8 - model creation
#   Ver 1.0.0 - 04.08.2023
#   ____ Simple Model ____
class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)

        self.regularization = nn.MSELoss() 

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

#   Step 9
#   Model training
input_size = X_train.shape[1]               #   Var number
model = FeedforwardNN(input_size)           #   Model init

#   Definicja funkcji kosztu i optymalizatora
#   Optimizer będzie używany do aktualizacji wag modelu, ADAM -> adaptive moment estimator
#   Model.parameter() przekazuje listę wszystkich parametrów | wag do optymalizatora, który będzie dostosowywał je w trakcie uczenia
#   LR -> learning rate, krok o który model będzie aktualizował wagi w kierunku minimalizacji funkcji straty
criterion = nn.MSELoss()                    #   Mean Squared Error, średnia kwadratów różnic między otrzymanymi wynikami, a prognozowanymi wynikami
optimizer = optim.Adam(model.parameters(), lr)  

#   We will check model predictions on test data without training
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss_pre = criterion(y_pred, y_test.unsqueeze(1))
    
    vis_list_test_pre = []
    for i in range(len(y_pred)):
        list = []
        list.append(y_pred[i][0])
        list.append(y_test[i])
        vis_list_test_pre.append(list)

#   Switch back to train model
model.train()

#   Step 10
#   ________ Training _______
#
start_time = time.time()
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1)) #   Do jednowymiarowego wektora outputu dodajemy dodatkowy wymiar
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   

    if epoch >= num_epochs - 1:
        #   Just visualize data
        vis_list = []
        for i in range(len(outputs)):
            list = []
            list.append(outputs[i][0])
            list.append(y_train[i])
            vis_list.append(list)

end_time = time.time()
execution_time = end_time - start_time

#   Step 1
#   _____ TESTOWANIE ____ 
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test.unsqueeze(1))
    
    vis_list_test = []
    for i in range(len(y_pred)):
        list = []
        list.append(y_pred[i][0])
        list.append(y_test[i])
        vis_list_test.append(list)

with open("results.txt", "w") as file:
    file.write(f"Learning time {execution_time} seconds\n")
    file.write(f"Epochs number {num_epochs}\n")
    file.write(f"Loss during last training epoch: {loss.item()}\n")
    file.write(f"Loss during test: {test_loss.item()}\n")
    file.write(f"Loss in pre test: {test_loss_pre.item()}\n")
    
    file.write(f"\nResults\nTraining data\t\tTest data\n")
    file.write("Prediction | Real | Prediction | Real | Before traing pred | Before train real\n")
    file.write("-" * 72 + "\n")
    
    for i in range(min(len(vis_list), len(vis_list_test))):
        prediction1, real1 = vis_list[i]
        prediction2, real2 = vis_list_test[i]
        prediction3, real3 = vis_list_test_pre[i]
        file.write(f"{prediction1:.4f}\t\t{real1:.4f}\t\t{prediction2:.4f}\t\t{real2:.4f}\t\t{prediction3:.4f}\t\t{real3:.4f}\n")
print("Wyniki zostały zapisane do pliku 'wyniki.txt'")
torch.save(model, 'model.pth')