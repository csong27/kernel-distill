from sklearn.metrics import euclidean_distances
import numpy as np


class SEiso(object):
    '''
    SEiso kernel function with hyper-parameters log_ell, log_sf
    k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    :param hyp:  hyper-parameters for SEiso kernel
    '''
    @staticmethod
    def evaluate(x, z, hyp):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)

        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])
        K = euclidean_distances(x / ell, z / ell, squared=True)   # (x-z)^T (x-z)
        return sf2 * np.exp(-K / 2)
    
    @staticmethod
    def gradient(x, z, hyp):
        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])
        K = euclidean_distances(x / ell, z / ell, squared=True)   # (x-z)^T (x-z)
        grad_log_ell = sf2 * np.exp(-K / 2) * K
        grad_log_sf = 2 * sf2 * np.exp(-K / 2) * K
        return grad_log_ell, grad_log_sf


if __name__ == '__main__':
    pass
