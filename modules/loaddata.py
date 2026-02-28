import os
import numpy as np

def load_response_functions(path, name):
    files = os.listdir(path)
    qvals = []
    data = []
    for file in files:
        q = file.split("@")[1].split(".txt")[0]
        if name in file:
            qvals.append(float(q))
            data.append(np.loadtxt(path + file))

    qvals = np.nan_to_num(np.array(qvals))
    data = np.nan_to_num(np.array(data))

    return qvals, data
