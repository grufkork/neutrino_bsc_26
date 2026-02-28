import numpy as np
import matplotlib.pyplot as plt

def peak(q, M):
    return np.sqrt(q**2+M**2)-M

def generate_colourscale(N):
    return plt.cm.jet(np.linspace(0,1,N))
