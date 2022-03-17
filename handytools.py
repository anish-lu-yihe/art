import numpy as np

def interp2coordinates(X, Y, N_interp):
    t = np.linspace(0, 1, N_interp + 2)
    z = [np.interp(t, [0, 1], [x, y]) for x, y in zip(X, Y)]
    return np.transpose(z)[1:-1]