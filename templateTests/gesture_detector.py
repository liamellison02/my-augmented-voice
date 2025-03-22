import csv
import ast
import numpy as np
import fastdtw 
from scipy.spatial.distance import euclidean

def load_time_series_from_csv(filepath):
    time_steps = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Each row is a time step; parse each cell
            flattened = []
            for cell in row:
                # Safely evaluate the tuple string into a Python tuple
                try:
                    vector = ast.literal_eval(cell)
                except Exception as e:
                    raise ValueError(f"Error parsing cell '{cell}': {e}")
                # If the evaluated cell is a tuple or list, extend; otherwise, add as single float
                if isinstance(vector, (list, tuple)):
                    flattened.extend(vector)
                else:
                    flattened.append(vector)
            time_steps.append(flattened)
    
    return np.array(time_steps)

def func_init(): 
    csv_files = [
    "output_csv/howareyou.csv", 
    "output_csv/nicetomeetu.csv", 
    ]
    arr = []
    for csv_file in csv_files: 
        D_entry = load_time_series_from_csv(csv_file)
        arr.append(D_entry)
    return arr 

def best_match_dtw(V, D):
    best_index = None
    best_distance = float('inf')
    for i, series in enumerate(D):
        distance, _ = fastdtw(V, series, dist=euclidean)
        if distance < best_distance:
            best_distance = distance
            best_index = i
    return best_index
    
if __name__ == '__main__':
    arr = func_init()
    # Code here     
