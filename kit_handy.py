import numpy as np
import sklearn.datasets as ds
import numpy.random as rnd


def load_data(dataset):
    if dataset == 'iris':
        iris = ds.load_iris()
        data = iris['data'] / np.max(iris['data'], axis=0)
        label = iris['target']
    elif dataset == 'gaussian2d':
        labels = [0, 1, 2]
        catnum = len(labels)
        mean = rnd.uniform(size=(catnum, 2))
        cov = [np.diag(d) for d in rnd.uniform(high=0.01, size=(catnum, 2))]
        catsize = 50
        label = np.repeat(labels, catsize)
        data = np.empty((0, 2))
        for m, c in zip(mean, cov):
            raw = np.random.multivariate_normal(m, c, size=catsize)
            pos = np.abs(raw)
            data = np.append(data, pos, axis=0)

    return data, label


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
