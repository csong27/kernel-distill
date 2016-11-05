import numpy as np
import scipy.sparse as sparse
from sklearn import linear_model
from kernel.seiso import SEiso
from backend.gpml import GPML
from optimizer import AdaGrad

np.random.seed(13423)


class Distillation(object):
    def __init__(self, kernel, X, U, num_iters, hyp, width, eta=1e-4, W=None, update_hyp=False, sparse_W=False):
        self.n = X.shape[0]
        self.m = U.shape[0]
        self.X = X
        self.U = U
        self.num_iters = num_iters
        self.hyp = hyp
        self.error = None
        self.eta = eta
        self.kernel = kernel
        self.update_hyp = update_hyp
        self.width = width
        # evaluate kernel
        self.KI = kernel.evaluate(U, U, hyp)
        self.dKI = kernel.gradient(U, U, hyp)
        self.K = kernel.evaluate(X, X, hyp)
        self.K_xu = kernel.evaluate(X, U, hyp)
        self.opt = AdaGrad(eta, (self.n, self.m))

        if W is None:
            self.W = np.zeros((self.n, self.m))
            self.regression_fit()
        else:
            self.W = W

        # set up the mask
        if sparse_W:
            self.W = sparse.csr_matrix(self.W)
            self.mask = sparse.csr_matrix(self.mask)

        self.sparse_W = sparse_W
        self.print_error()

    @staticmethod
    def find_neighbors(x, U, width):
        dif = x - U
        norms = np.apply_along_axis(np.linalg.norm, 1, dif)
        ind = np.argsort(norms)[: width]
        return ind

    def regression_fit(self):
        for i in xrange(self.n):
            ind = self.find_neighbors(self.X[i], self.U, self.width)
            regr = linear_model.LinearRegression()
            regr.fit(self.KI[ind].T, self.K_xu[i])
            self.W[i, ind] = regr.coef_
        self.mask = np.zeros((self.n, self.m))
        self.mask[self.W != 0] = 1.

    def grad_update_W(self, row_ind):
        # grad = np.zeros(self.W[row_ind].shape)
        # for i in xrange(self.n):
        #     if i == row_ind:
        #         grad += 4 * (reduce(np.dot, [self.W[row_ind], self.KI, self.W[i]]) - self.K[row_ind, i])\
        #                 * np.dot(self.KI, self.W[i])
        #     else:
        #         grad += 2 * (reduce(np.dot, [self.W[row_ind], self.KI, self.W[i]]) - self.K[row_ind, i])\
        #                 * np.dot(self.KI, self.W[i])
        if self.sparse_W:
            A = self.W[row_ind].dot(self.KI)
            A = self.W.dot(A.T).flatten() - self.K[row_ind]
            A *= 2
        else:
            A = 2 * (reduce(np.dot, [self.W[row_ind], self.KI, self.W.T]) - self.K[row_ind])
        A[row_ind] *= 2
        if self.sparse_W:
            B = self.W.dot(self.KI).T
            grad = B.dot(A)
        else:
            B = np.dot(self.KI, self.W.T)
            grad = np.dot(B, A)
        return grad

    def grad_update_hyp(self, eta):
        diff_K = self.diff_to_K()
        for i in range(len(self.hyp)):
            if self.sparse_W:
                W_dKI = self.W.dot(self.dKI[:, :, i]).T
                grad = 2 * np.sum(diff_K * self.W.dot(W_dKI))
            else:
                grad = 2 * np.sum(diff_K * reduce(np.dot, [self.W, self.dKI[:, :, i], self.W.T]))
            self.hyp[i] -= eta / 1e3 * grad
        # update kernel
        self.dKI = self.kernel.gradient(self.U, self.U, self.hyp)
        self.KI = self.kernel.evaluate(self.U, self.U, self.hyp)

    def grad_descent(self):
        for _ in xrange(self.num_iters):
            grad = np.zeros(self.W.shape)
            for i in xrange(self.n):
                grad_i = self.grad_update_W(i)
                grad[i] = grad_i.flatten()
            # adjust gradient by optimizer
            grad = self.opt.adjust_grad(grad)
            # project to the banded matrix space
            if self.sparse_W:
                grad = self.mask.multiply(grad)
            else:
                grad = grad * self.mask

            self.W -= grad
            if self.update_hyp:
                self.grad_update_hyp(self.eta)
            self.print_error()

    def diff_to_K(self):
        if self.sparse_W:
            W_KI = self.W.dot(self.KI).T
            return self.W.dot(W_KI) - self.K
        else:
            return reduce(np.dot, [self.W, self.KI, self.W.T]) - self.K

    def print_error(self):
        self.error = np.linalg.norm(self.diff_to_K())
        print self.error, np.linalg.norm(self.K)


def test():
    n = 500
    dim = 2
    k = 15
    m = k ** dim

    X = np.random.normal(0, 10, size=(n, dim))
    X = np.sort(X, axis=0)
    inducing = np.random.choice(n, m, replace=False)
    U = X[inducing]
    hyp = [np.log(2), np.log(2)]
    kernel = SEiso()

    # check kiss
    gpml = GPML()
    cov = 'covSEiso'
    K = gpml.cov_eval(X, cov, hyp)
    xg, row, col, val, dval, N = gpml.interpolate_grid(X, 1, k, expand_grid=True)
    print xg.shape

    W = sparse.csr_matrix((val, (row, col)), shape=(n, N)).toarray()
    Kuu = kernel.evaluate(xg, xg, hyp)
    K_approx = reduce(np.dot, [W, Kuu, W.T])
    print np.linalg.norm(K - K_approx)
    # distillation
    distill = Distillation(kernel=kernel, X=X, U=U, width=5, hyp=hyp, num_iters=1000, eta=1e-2, sparse_W=True,
                           update_hyp=False)
    distill.grad_descent()


if __name__ == '__main__':
    test()
