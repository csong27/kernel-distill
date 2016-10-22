import numpy as np
from scipy.linalg import cholesky, cho_solve


class ExactInference(object):
    def __init__(self, hyp, cov):
        self.cov_hyp = hyp[:-1]
        self.log_noise = hyp[-1]
        self.cov = cov(self.cov_hyp)

    def setup(self, x, y):
        n = x.shape[0]
        sn2 = np.exp(2 * self.log_noise)
        K = self.cov.evaluate(x, x)
        L = cholesky(K + sn2 * np.eye(n), lower=True)
        alpha = cho_solve((L, True), y)

        return n, sn2, K, L, alpha

    def nlml(self, x, y):
        n, sn2, K, L, alpha = self.setup(x, y)
        y_term = np.dot(y, alpha) / 2
        log_det_term = np.sum(np.log(np.diag(L)))
        pi_term = n * np.log(np.pi * 2) / 2
        return y_term + log_det_term + pi_term

    def dnlml(self, x, y):
        n, sn2, K, L, alpha = self.setup(x, y)
        Q = cho_solve((L, True), np.eye(n)) - np.dot(alpha, alpha)
        grad = []
        grad_cov_hyp = self.cov.gradient(x, x)
        for g in grad_cov_hyp:
            grad.append(np.trace(np.dot(Q, g)) / 2)
        grad.append(sn2 * np.trace(Q))
        return grad
