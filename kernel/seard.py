from sklearn.metrics import euclidean_distances
import numpy as np


class SEard(object):
    '''
    SEard kernel function with hyper-parameters log_ell, log_sf
    k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    :param hyp:  hyper-parameters for SEiso kernel
    '''

    @staticmethod
    def evaluate(x, z, hyp):
        ell = np.exp(hyp[:-1])
        sf2 = np.exp(2 * hyp[-1])
        diag_ell = np.diag(1. / ell)
        K = euclidean_distances(np.dot(x, diag_ell), np.dot(z, diag_ell), squared=True)  # (x-z)^T (x-z)
        return sf2 * np.exp(-K / 2)

    @staticmethod
    def gradient(x, z, hyp):
        ell = np.exp(hyp[:-1])
        sf2 = np.exp(2 * hyp[-1])
        diag_ell = np.diag(1. / ell)
        K = euclidean_distances(np.dot(x, diag_ell), np.dot(z, diag_ell), squared=True)  # (x-z)^T (x-z)
        return sf2 * np.exp(-K / 2)


if __name__ == '__main__':
    x = np.random.normal(size=(5, 2))
    k = SEard.evaluate(x, x, [np.log(2)] * 3)
    from gpml_kernel import GPMLKernel
    kernel = GPMLKernel('covSEard')
    k0 = kernel.evaluate(x, x, [np.log(2)] * 3)
    print k - k0
