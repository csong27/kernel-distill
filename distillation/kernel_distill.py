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
                 use_kmeans=False, optimizer='sgd', K=None):
        self.n = X.shape[0]
        self.m = U.shape[0]
        self.X = X
        self.y = y
        self.U = self.inducing_kmeans() if use_kmeans else U
        self.num_iters = int(num_iters)
        self.hyp = hyp
        self.error = None
        self.eta = eta
        self.kernel = kernel
        self.update_hyp = update_hyp
        self.width = width
        self.sigmasq = sigmasq
        self.pre_mean = None
        self.pre_var = None
        self.pre_alpha = None

        # evaluate kernel
        print 'Evaluating kernel matrix...'
        self.KI = kernel.evaluate(self.U, self.U, self.hyp)
        self.dKI = kernel.gradient(self.U, self.U, self.hyp) if self.update_hyp else None
        self.K = kernel.evaluate(self.X, self.X, self.hyp) if K is None else K
        self.K_xu = kernel.evaluate(self.X, self.U, self.hyp)

        # setup optimizer
        if optimizer == 'adagrad':
            self.opt = AdaGrad(eta, (self.n, self.m))
        elif optimizer == 'sgd':
            self.opt = SGD(eta)
        else:
            raise ValueError(optimizer)

        # store inducing points
        self.kd_tree = spatial.cKDTree(self.U)

        # initialize W
        if W is None:
            self.W = np.zeros((self.n, self.m))
            self.regression_fit()
        else:
            self.W = W

        # set up the mask
        self.mask = np.zeros((self.n, self.m))
        self.mask[self.W != 0] = 1.

        # use sparse representation
        self.W = sparse.csr_matrix(self.W)
        self.mask = sparse.csr_matrix(self.mask)

        # initial error
        self.print_error()

    def inducing_kmeans(self):
        print 'Using K-means to choose {} inducing points...'.format(self.m)
        kmeans = KMeans(n_clusters=self.m, verbose=0)
        kmeans.fit(self.X)
        return kmeans.cluster_centers_

    def nearest_neighbor1(self, x, width=None):
        if width is None:
            width = self.width

        if len(x.shape) == 2:
            x = x.flatten()

        d = x.shape[0]
        ind = []
        for dim in xrange(d):
            d_val = x[dim]
            u_vals = self.U[:, dim]
            distance = np.square(u_vals - d_val)
            ind.append(np.argsort(distance)[:width])
        ind = np.concatenate(ind)
        return np.unique(ind)

    def nearest_neighbor(self, x, width=None):
        if width is None:
            width = self.width
        _, ind = self.kd_tree.query(x, width)
        return ind.flatten()

    def regression_fit(self):
        print 'Using linear regression as initialization...'
        for i in xrange(self.n):
            ind = self.nearest_neighbor(self.X[i])
            regr = linear_model.LinearRegression()
            regr.fit(self.KI[ind].T, self.K_xu[i])
            self.W[i, ind] = regr.coef_

    def grad_update_W(self):
        B = self.W.dot(self.KI).T
        A = self.diff_to_K()
        di = np.diag_indices(self.n)
        A[di] *= 2
        grad = B.dot(A)
        return grad.T

    def grad_update_hyp(self, eta):
        diff_K = self.diff_to_K()
        for i in range(len(self.hyp)):
            W_dKI = self.W.dot(self.dKI[:, :, i]).T
            grad = 2 * np.sum(diff_K * self.W.dot(W_dKI))
            self.hyp[i] -= eta / 1e3 * grad
        # update kernel
        self.dKI = self.kernel.gradient(self.U, self.U, self.hyp)
        self.KI = self.kernel.evaluate(self.U, self.U, self.hyp)

    def grad_descent(self):
        print 'Starting distillation...'
        for it in xrange(self.num_iters):
            # grad = np.zeros(self.W.shape)
            print 'At iteration {}'.format(it)
            # adjust gradient by optimizer
            grad = self.opt.adjust_grad(self.grad_update_W())
            # project to the banded matrix space
            grad = self.mask.multiply(grad)
            self.W -= grad
            if self.update_hyp:
                self.grad_update_hyp(self.eta)
            self.print_error()

    def precompute_alpha(self, L):
        self.pre_alpha = sl.cho_solve((L, True), self.y)

    def precompute_mean(self, L):
        print 'Precompute for mean...'
        b = self.W.dot(self.KI).T
        if self.pre_alpha is None:
            self.precompute_alpha(L)
        self.pre_mean = np.dot(b, self.pre_alpha)

    def precompute_var(self, L):
        print 'Precompute for variance...'
        b = self.W.dot(self.KI)
        self.pre_var = np.dot(b.T, sl.cho_solve((L, True), b))
        self.pre_var = np.diag(self.pre_var)

    # def kiss_operator(self, vec):
    #     kiss_vec = self.W.dot(self.KI.dot(self.W.T.dot(vec)))
    #     kiss_vec += self.sigmasq * vec
    #     return kiss_vec
    #
    # def cg_solve(self, b, n_iter):
    #     A = sparse.linalg.LinearOperator((self.n, self.n), matvec=self.kiss_operator)
    #     val, err = sparse.linalg.cg(A, b, maxiter=n_iter)
    #     if err < 0:
    #         raise ValueError('CG failed.')
    #     return val

    def predict(self, x_star, width=None, sample_ratio=2):
        ind = self.nearest_neighbor(x_star, width)
        W_star = self.get_W_star(x_star, ind, sample_ratio)
        # predicted mean
        pred_mean = W_star.dot(self.pre_mean)
        # predicted variance
        K_xstar_xstar = self.kernel.evaluate(x_star, x_star, self.hyp)[0][0]
        explained_var = W_star.dot(self.pre_var)
        pred_var = K_xstar_xstar - explained_var + self.sigmasq
        return pred_mean, pred_var

    def get_W_star(self, x_star, ind, sample_ratio=2):
        sampled_ind = self.nearest_neighbor(x_star, self.width * sample_ratio)
        K_xstar_u = self.kernel.evaluate(x_star, self.U[sampled_ind], hyp=self.hyp)
        reg = linear_model.LinearRegression()
        reg.fit(self.KI[ind].T[sampled_ind], K_xstar_u.flatten())
        W_star = sparse.csr_matrix((reg.coef_, (np.zeros_like(ind), ind)), shape=(1, self.m))
        return W_star

    def diff_to_K(self):
        return self.approx_K() - self.K

    def approx_K(self):
        W_KI = self.W.dot(self.KI).T
        app_K = self.W.dot(W_KI)
        return app_K

    def print_error(self):
        self.error = np.linalg.norm(self.diff_to_K())
        print 'Error norm {}, True K norm {}'.format(self.error, np.linalg.norm(self.K))

    def precompute(self, use_true_K=False):
        K = self.K if use_true_K else self.approx_K()
        K += self.sigmasq * np.eye(self.n)
        L = sl.cholesky(K, lower=True)
        self.precompute_mean(L)
        self.precompute_var(L)


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
