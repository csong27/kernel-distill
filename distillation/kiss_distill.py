import numpy as np
import scipy.sparse as sparse
from sklearn import linear_model
from functools import reduce
from kernel.seiso import SEiso
from backend.gpml import GPML


np.random.seed(13423)


class Distillation(object):
    def __init__(self, kernel, X, U, num_iters, hyp, width, eta=1e-4):
        self.n, _ = X.shape
        self.m, _ = U.shape
        self.X = X
        self.U = U
        self.num_iters = num_iters
        self.W = np.zeros((self.n, self.m))
        self.difK = np.zeros((self.n, self.n))
        self.hyp = hyp
        self.error = None
        self.banded_mask = np.zeros((self.n, self.m))
        self.eta = eta
        self.kernel = kernel

        # evaluate kernel
        self.KI = kernel.evaluate(U, U, hyp)
        self.dKI = kernel.gradient(U, U, hyp)
        self.K = kernel.evaluate(X, X, hyp)
        self.K_xu = kernel.evaluate(X, U, hyp)
        self.regress_init(width)

    @staticmethod
    def find_neighbors(x, U, width):
        dif = x - U
        norms = np.apply_along_axis(np.linalg.norm, 1, dif)
        ind = np.argsort(norms)[: width]
        return ind

    def regress_init(self, width):
        for i in xrange(self.n):
            ind = self.find_neighbors(self.X[i], self.U, width)
            regr = linear_model.LinearRegression()
            regr.fit(self.KI[ind].T, self.K_xu[i])
            self.W[i, ind] = regr.coef_
        self.banded_mask[self.W != 0] = 1.
        self.print_error()

    def grad_update_W(self, row_ind, eta):
        grad = np.zeros(self.W[row_ind, :].shape)
        for i in xrange(self.n):
            if i == row_ind:
                grad += 4 * (reduce(np.dot, [self.W[row_ind], self.KI, self.W[i]]) - self.K[row_ind, i]) * np.dot(self.KI, self.W[i])
            else:
                grad += 2 * (reduce(np.dot, [self.W[row_ind], self.KI, self.W[i]]) - self.K[row_ind, i]) * np.dot(self.KI, self.W[i])
        self.W[row_ind] -= eta * grad

    def grad_update_hyp(self, eta):
        for i in range(len(self.hyp)):
            grad = 2 * np.sum(self.difK * reduce(np.dot, [self.W, self.dKI[:, :, i], self.W.T]))
            self.hyp[i] = self.hyp[i] - eta * grad
        # update kernel
        self.dKI = self.kernel.gradient(self.U, self.U, self.hyp)
        self.KI = self.kernel.evaluate(self.U, self.U, self.hyp)

    def grad_descent(self):
        for _ in xrange(self.num_iters):
            for i in xrange(self.n):
                self.grad_update_W(i, self.eta)
                # project to the banded matrix space
                self.W = self.W * self.banded_mask
            # self.grad_update_hyp(self.eta)
            self.print_error()

    def print_error(self):
        self.difK = reduce(np.dot, [self.W, self.KI, self.W.T]) - self.K
        self.error = np.linalg.norm(self.difK)
        print self.error, np.linalg.norm(self.K)


def test():
    n = 100
    m = 20
    X = np.random.normal(0, 5, size=(n, 1))
    X = np.sort(X, axis=0)
    inducing = np.random.choice(n, m, replace=False)
    U = X[inducing]
    hyp = [np.log(2), np.log(2)]
    kernel = SEiso()

    # check kiss
    gpml = GPML()
    cov = 'covSEiso'
    K = gpml.cov_eval(X, cov, hyp)
    xg, row, col, val, dval, N = gpml.interpolate_grid(X, 1, float(m))
    W = sparse.csr_matrix((val, (row, col)), shape=(n, N)).toarray()
    Kuu = kernel.evaluate(xg, xg, hyp)
    K_approx = reduce(np.dot, [W, Kuu, W.T])
    print np.linalg.norm(K - K_approx)
    # distillation
    distill = Distillation(kernel=kernel, X=X, U=U, width=5, hyp=hyp, num_iters=1000, eta=1e-4)
    distill.grad_descent()


if __name__ == '__main__':
    test()
