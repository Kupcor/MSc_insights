import re
import math
import numpy as np

def extract_values(file_path):
    predicted_values = []
    actual_values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if i == 0:
                i += 1
                continue
            match = re.search(r'Predicted: ([-\d.]+) +\| Actual: ([-\d.]+)', line)
            if match:
                predicted_values.append(float(match.group(1)))
                actual_values.append(float(match.group(2)))

    return predicted_values, actual_values

file_path = 'test_data/model_v3.py_test_results_data.txt'
predicted_values, actual_values = extract_values(file_path)

sum = 0
for i in range(len(predicted_values)):
    #sum += abs(predicted_values[i] - actual_values[i])                 # L1
    sum += math.pow(predicted_values[i] - actual_values[i], 2)          # L2

print(sum / len(predicted_values))
total_loss = np.mean(np.square(np.array(predicted_values) - np.array(actual_values)))
print(total_loss)