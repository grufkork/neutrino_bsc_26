import os
import numpy as np

# Data is structured as
#   [q_value][curve_index(0-4)][w=0|y=1][point_index]

def load_response_functions(path, name):
    files = os.listdir(path)
    qvals = []
    data = []
    for file in files:
        q = file.split("@")[1].split(".txt")[0]
        if name in file:
            qvals.append(float(q))
            raw = np.loadtxt(path + file)

            qdata = []
            
            w = raw[:,0]
            for i in range(1, raw.shape[1]):
                qdata.append([w, raw[:,i]])
            data.append(qdata)

    qvals = np.nan_to_num(np.array(qvals))
    data = np.nan_to_num(np.array(data))

    sorted_indices = np.argsort(qvals)
    qvals = qvals[sorted_indices]
    data = data[sorted_indices]

    return qvals, data

def load_ab_initio(path):
    curvenames = ["Rxy", "R00", "Rt"]

    data = []

    qvals = range(50,450, 50)
    for q in qvals:
        for curve in curvenames:
            filename = f"CR_q{q}_{curve}_NNLO_GO_450.dat"
            raw = np.loadtxt(path + filename)
        w = raw[:,0]
        y = raw[:,1]
        data.append([w, y])

    qvals = np.nan_to_num(np.array(qvals))
    data = np.nan_to_num(np.array(data))

    sorted_indices = np.argsort(qvals)
    qvals = qvals[sorted_indices]
    data = data[sorted_indices]

    return qvals, data