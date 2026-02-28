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
    non_zero_indices = np.argwhere(y > 0)
    if len(non_zero_indices) == 0:
        non_zero_indices = [[0]]
    lower = x[non_zero_indices[0][0]]
    upper = x[non_zero_indices[-1][0]]


    return lower, upper

def get_peak_width(x, y):
    lower, upper = get_peak_bound_indices(x, y)
    return (upper-lower)
