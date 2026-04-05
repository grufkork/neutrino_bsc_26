import os
import numpy as np

# Data is structured as
#   [q_value][curve_index(0-4)][w=0|y=1][point_index]

LFG_CURVENAMES = ["R00", "R0z", "Rzz", "Rxx", "Rxy"]

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
            
            w = raw[:,0]*1000 # Convert to MeV
            for i in range(1, raw.shape[1]):
                qdata.append([w, raw[:,i]/1000]) # Tables are in units of 1/GeV
            data.append(qdata)

    qvals = np.nan_to_num(np.array(qvals)) * 1000 # Convert to MeV
    data = np.nan_to_num(np.array(data))

    sorted_indices = np.argsort(qvals)
    qvals = qvals[sorted_indices]
    data = data[sorted_indices]

    return qvals, data


ab_initio_curvenames = ["Rxy", "R00", "Rt", "R0z", "Rzz"]

def load_ab_initio(path):

    data = []

    qvals = range(50,450, 50)
    for q in qvals:
        qdata = []
        for curve in ab_initio_curvenames:
            filename = f"CR_q{q}_{curve}_NNLO_GO_450.dat"
            raw = np.loadtxt(path + filename)
            w = raw[:,0]
            avg = (raw[:,1] + raw[:,2])/2 * 1000 # Average
            y1 = (raw[:,1])*1000 # lower 
            y2 = (raw[:,2])*1000 # upper
            qdata.append([w, avg, y1, y2])
        data.append(qdata)

    qvals = np.nan_to_num(np.array(qvals))
    data = np.nan_to_num(np.array(data))

    sorted_indices = np.argsort(qvals)
    qvals = qvals[sorted_indices]
    data = data[sorted_indices]

    return qvals, data

def load_spectral(path):
    curvenames = ["Rxy", "R00", "Rxx"]

    data = []

    qvals = np.arange(0.05, 0.45, 0.025)
    for q in qvals:
        qdata = []
        for curve in curvenames:
            q = round(q, 4)
            filename = f"{curve}_q_{q}.txt"
            raw = np.loadtxt(path + filename)
            w = raw[:,0]*1000
            y = raw[:,1]
            qdata.append([w, y])
        data.append(qdata)

    qvals = np.nan_to_num(np.array(qvals))*1000
    data = np.nan_to_num(np.array(data))

    sorted_indices = np.argsort(qvals)
    qvals = qvals[sorted_indices]
    data = data[sorted_indices]

    return qvals, data

def build_spectral_interpolator(spectral_q, spectral_vals):
    from scipy.interpolate import CloughTocher2DInterpolator

    def build_spectral_interpolator(q_vals, vals, curve_index):
        wvals = vals[:, curve_index, 0, :]
        yvals = vals[:, curve_index, 1, :]

        points = []
        vals = []
        for i in range(len(q_vals)):
            for j in range(wvals.shape[1]):
                points.append((q_vals[i], wvals[i, j]))
                vals.append(yvals[i, j])

        interpolator_function = CloughTocher2DInterpolator(
            points,
            values=vals,
            # fill_value=0.0, # Need to trim training data to not fall outside this 
            # method="cubic",
            # bounds_error=True,
            # fill_value=np.nan,
        )
        return interpolator_function

    spectral_functions = []
    for i in range(spectral_vals.shape[1]):
        f = build_spectral_interpolator(spectral_q, spectral_vals, i)
        spectral_functions.append(f)
    return spectral_functions

def load_t2k(path):
    bins = []
    vals = []
    with open(path, "r") as f:
        lines = f.readlines()[3:]
    for line in lines:
        line = line.strip()
        parts = line.split()
        i = float(parts[0])
        bin_lower = float(parts[1])
        bin_upper = float(parts[3])
        flux = float(parts[4])
        bins.append((bin_lower, bin_upper))
        vals.append(flux)
    return np.array(bins)*1000, np.array(vals)