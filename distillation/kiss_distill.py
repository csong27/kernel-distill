import numpy as np
import scipy.linalg as sl
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cluster import KMeans
from kernel.seiso import SEiso
from backend.gpml import GPML
from optimizer import AdaGrad, SGD


np.random.seed(13423)


class Distillation(object):
    def __init__(self, kernel, X, y, U, num_iters, hyp, sigmasq, width, eta=1e-4, W=None, update_hyp=False,
                 sparse_W=True, use_kmeans=True):
        self.n = X.shape[0]
        self.m = U.shape[0]
        self.X = X
        self.U = self.inducing_kmeans() if use_kmeans else U
        self.num_iters = num_iters
        self.hyp = hyp
        self.error = None
        self.eta = eta
        self.kernel = kernel
        self.update_hyp = update_hyp
        self.width = width
        self.sigmasq = sigmasq
        self.pre_mean = None
        self.pre_var = None
        self.y = y
        # evaluate kernel
        self.KI = kernel.evaluate(U, U, hyp)
        self.dKI = kernel.gradient(U, U, hyp)
        self.K = kernel.evaluate(X, X, hyp)
        self.K_xu = kernel.evaluate(X, U, hyp)
        self.opt = AdaGrad(eta, (self.n, self.m))
        self.kd_tree = spatial.cKDTree(self.U)

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

    def inducing_kmeans(self):
        kmeans = KMeans(n_clusters=self.m)
        kmeans.fit(self.X)
        return kmeans.cluster_centers_

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

    def precompute_mean(self):
        self.pre_mean = self.cg_solve(self.y, n_iter=self.n)

    def precompute_var(self):
        pass

    def kiss_operator(self, vec):
        kiss_vec = self.W.dot(self.KI.dot(self.W.T.dot(vec)))
        kiss_vec += self.sigmasq * vec
        return kiss_vec

    def cg_solve(self, b, n_iter):
        A = sparse.linalg.LinearOperator((self.n, self.n), matvec=self.kiss_operator)
        val, err = sparse.linalg.cg(A, b, maxiter=n_iter)
        if err < 0:
            raise ValueError('CG failed.')
        return val

    def predict_mean(self, x_star, width=None, use_exact=True):
        if self.pre_mean is None:
            self.precompute_mean()
        if width is None:
            width = self.width
        _, ind = self.kd_tree.query(x_star, width, 0.)
        K_xstar_x = self.get_K_xstar_x(x_star, ind, use_exact)
        return np.dot(K_xstar_x, self.pre_mean)

    def predict_variance(self, x_star, width=None, use_exact=True):
        if width is None:
            width = self.width
        _, ind = self.kd_tree.query(x_star, width, 0.)
        K_xstar_x = self.get_K_xstar_x(x_star, ind, use_exact)
        K_xstar_xstar = self.kernel.evaluate(x_star, x_star, self.hyp)[0][0]
        val = self.cg_solve(K_xstar_x, n_iter=500)
        return K_xstar_xstar - np.dot(K_xstar_x, val)

    def get_W_star(self, x_star, ind=None, sample_size=20):
        sample_ind = np.random.choice(self.m, sample_size, replace=False)
        K_xstar_u = self.kernel.evaluate(x_star, self.U, hyp=self.hyp)
        reg = linear_model.LinearRegression()
        reg.fit(self.KI[ind].T[sample_ind], K_xstar_u.flatten()[sample_ind])
        W_star = sparse.csr_matrix((self.m, ))
        W_star[ind] = reg.coef_
        return W_star

    def get_K_xstar_x(self, x_star, ind=None, use_exact=False, sample_size=20):
        if use_exact:
            return self.kernel.evaluate(x_star, self.X, hyp=self.hyp)
        else:
            W = self.W
            if self.sparse_W:
                W = W.toarray()
            sample_ind = np.random.choice(self.m, sample_size, replace=False)
            K_xstar_u = self.kernel.evaluate(x_star, self.U, hyp=self.hyp)
            reg = linear_model.LinearRegression()
            reg.fit(self.KI[ind].T[sample_ind], K_xstar_u.flatten()[sample_ind])
            W_star = np.zeros(self.m)
            W_star[ind] = reg.coef_
            approx_K_xstar_x = reduce(np.dot, (W_star, self.KI, W.T))
            # print approx_K_xstar_x.shape
            return approx_K_xstar_x

    def diff_to_K(self):
        return self.approx_K() - self.K

    def approx_K(self):
        if self.sparse_W:
            W_KI = self.W.dot(self.KI).T
            return self.W.dot(W_KI)
        else:
            return reduce(np.dot, [self.W, self.KI, self.W.T])

    def print_error(self):
        self.error = np.linalg.norm(self.diff_to_K())
        print self.error, np.linalg.norm(self.K)


def test():
    n = 1000
    dim = 1
    k = 150
    m = 150

    X = np.random.normal(0, 5, size=(n, dim))
    X = np.sort(X, axis=0)
    inducing = np.random.choice(n, m, replace=False)
    inducing = np.sort(inducing)
    U = X[inducing]
    hyp = [np.log(9), np.log(1)]
    kernel = SEiso()

    # check kiss
    gpml = GPML()
    cov = 'covSEiso'
    K = gpml.cov_eval(X, cov, hyp)
    xg, row, col, val, dval, N = gpml.interpolate_grid(X, 1, k, expand_grid=True)
    W = sparse.csr_matrix((val, (row, col)), shape=(n, N)).toarray()
    Kuu = kernel.evaluate(xg, xg, hyp)
    K_approx = reduce(np.dot, [W, Kuu, W.T])
    print np.linalg.norm(K - K_approx)
    plt.imshow(np.abs(K - K_approx))
    plt.colorbar()
    plt.show()

    # distillation
    # distill = Distillation(kernel=kernel, X=X, U=U, width=16, hyp=hyp, num_iters=0, eta=1e-5, sparse_W=True,
    #                        update_hyp=False, W=None)
    # distill.grad_descent()
    # W1 = distill.W.toarray()
    # K_approx1 = reduce(np.dot, [W1, distill.Kuu, W1.T])
    #
    # plt.imshow(np.abs(K - K_approx1))
    # plt.colorbar()
    # plt.show()

    K_uu = kernel.evaluate(xg, xg, hyp)
    K_xu = kernel.evaluate(X, xg, hyp)

    L_uu = sl.cholesky(K_uu + 1e-6 * np.eye(m), lower=True)
    K_uu_inv = sl.cho_solve((L_uu, True), np.eye(m))

    K_sor = reduce(np.dot, [K_xu, K_uu_inv, K_xu.T])
    K_diag = np.diag(K - K_sor)
    K_fitc = K_sor + np.diag(K_diag)
    print np.linalg.norm(K - K_sor)
    print np.linalg.norm(K - K_fitc)

    plt.imshow(np.abs(K - K_fitc))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test()
