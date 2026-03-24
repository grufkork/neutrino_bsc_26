import numpy as np
import matplotlib.pyplot as plt

def peak(q, M):
    return np.sqrt(q**2+M**2)-M

def generate_colourscale(N):
    return plt.cm.jet(np.linspace(0,1,N))


def gaussian(x, amplitude, x0, sigma):
    return amplitude * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gaussian_fit(x, y):
    from scipy.optimize import curve_fit

    initial_guess = [np.max(y), x[np.argmax(y)], 1.0]
    try:
        params, _ = curve_fit(gaussian, x, y, p0=initial_guess, maxfev=10000)

    except RuntimeError:
        params = initial_guess

    return params