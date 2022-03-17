import numpy as np

def interp2coordinates(X, Y, N_interp):
    t = np.linspace(0, 1, N_interp + 2)
    z = [np.interp(t, [0, 1], [x, y]) for x, y in zip(X, Y)]
    return np.transpose(z)[1:-1]

def savefigure_datetime(figure, simname, dirname):
    from datetime import datetime
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    filename = '{}/{}_{}'.format(dirname, now_str, simname)
    print("figure saved at", filename)
    figure.savefig(filename)