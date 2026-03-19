import numpy as np 

import modules.utils 

def transform_curve(w, y, q):
    w_peak = modules.utils.peak(q, 1)
    w = (w-w_peak)/q
    return w, y

def transform_curve_inverse(w, y, q):
    w_peak = modules.utils.peak(q, 1)
    return w*q + w_peak, y

def curve_interpolate(c1_w, c1_y, c2_w, c2_y, q1, q2, q_out):
    # plt.plot((w - (peak_w))/q, R_L/amp_comp_func(q), label=q, c=colors[i])
    M = 0.85
    c1_w, c1_y = transform_curve(c1_w, c1_y, q1)
    c2_w, c2_y = transform_curve(c2_w, c2_y, q2)
    c2_y_interp = np.interp(c1_w, c2_w, c2_y) # Match functions on the same w points

    interp_factor = (q_out - q1)/(q2-q1)

    y_out = (c1_y * (1 - interp_factor) + c2_y_interp * interp_factor )
    x_out = (c1_w * (1 - interp_factor) + c2_w * interp_factor) # not sure if valid


    x_out, yout = transform_curve_inverse(x_out, y_out, q_out)[0]
    return x_out, y_out


def get_peak_bound_indices(x, y):
    # Get indices of non-zero block in middle of data
    avgheight = np.mean(y)
    non_zero_indices = np.argwhere(y > avgheight)
    if len(non_zero_indices) == 0:
        non_zero_indices = [[0]]
    lower = x[non_zero_indices[0][0]]
    upper = x[non_zero_indices[-1][0]]


    return lower, upper

def get_peak_width(x, y):
    lower, upper = get_peak_bound_indices(x, y)
    return (upper-lower)

def get_peak_width_gaussian(x, y):
    params = modules.utils.gaussian_fit(x, y)
    _, _, sigma = params
    return np.abs(sigma)

def extract_interp_param_curves(q, w, y):
    widths = []
    heights = []
    for qval, wval, val in zip(q, w, y):
        width = modules.interpolation_preprocess.get_peak_width_gaussian(wval, val)

        widths.append(width)
        heights.append(np.max(val))
    
    return widths, heights

def get_transform_params(q, data):
    widths_per_curve = []
    heights_per_curve = []

    for idx in range(len(data[0])):
        widths, heights = extract_interp_param_curves(q, data[:, idx, 0], data[:, idx, 1])
        widths_per_curve.append(widths)
        heights_per_curve.append(heights)
    
    return q, widths_per_curve, heights_per_curve

def transform_data(q, data, transform_params):
    q = np.copy(q)
    data_new = np.copy(data)
    q_source, widths_per_curve, heights_per_curve = transform_params
    
    for curve_idx in range(len(data[0])):
        for (i, (qval, wvals, yvals)) in enumerate(zip(q, data[:,curve_idx,0], data[:,curve_idx,1])):
            estimated_width = np.interp(qval, q_source, widths_per_curve[curve_idx])
            estimated_height = np.interp(qval, q_source, heights_per_curve[curve_idx])

            wpeak = modules.utils.peak(qval, 1)
            wvals = wvals - wpeak
            wvals = wvals / estimated_width
            yvals = yvals / estimated_height

            data_new[i,curve_idx,0] = wvals
            data_new[i,curve_idx,1] = yvals
    
    return data_new