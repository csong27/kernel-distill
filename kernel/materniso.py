from sklearn.metrics import euclidean_distances
import numpy as np


class Materniso(object):
    '''
    Materniso kernel function with hyperparameters log_ell, log_sf
    k(x^p,x^q) = sf^2 * f( sqrt(d)*r ) * exp(sqrt(d)*r)
    :param hyp:  hyperparameters for Materniso kernel
    '''
    def __init__(self, d):
        assert d in [1, 3, 5]
        if d == 1:
            self.f = lambda t: 1.
            self.df = lambda t: 1.
        elif d == 3:
            self.f = lambda t: 1. + t
            self.df = lambda t: t
        else:
            self.f = lambda t: 1. + t * (1 + t / 3)
            self.df = lambda t: t * (1 + t) / 3

        self.m = lambda t: self.f(t) * np.exp(t)
        self.dm = lambda t: self.df(t) * np.exp(t)

        self.d = d

    def evaluate(self, x, z, hyp):
        if len(x.shape) == 1:
            x = x.reshape(1, 1)
        if len(z.shape) == 1:
            z = z.reshape(1, 1)

        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])

        K = euclidean_distances(np.sqrt(self.d) * x / ell, np.sqrt(self.d) * z / ell)
        return sf2 * self.m(K)

    def gradient(self, x, z, hyp):
        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])
        K = euclidean_distances(np.sqrt(self.d) * x / ell, np.sqrt(self.d) * z / ell)
        grad_log_ell = sf2 * self.dm(K)
        grad_log_sf = 2 * sf2 * self.dm(K)
        shape = (K.shape[0], -1, 1)
        return np.concatenate((grad_log_ell.reshape(shape), grad_log_sf.reshape(shape)), axis=2)


if __name__ == '__main__':
    pass
