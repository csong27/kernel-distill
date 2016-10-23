from cov import SEiso, Materniso
from matrix_comp import MatrixCompletion
from scipy.linalg import cholesky, cho_solve
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123132)


class KernelApproximate(object):
    def __init__(self, kernel, x, num_samples, inducing_points, hyp):
        self.kernel = kernel
        self.x = x
        self.num_samples = num_samples
        self.Omega = {}
        self.values = {}
        self.inducing_points = inducing_points
        self.hyp = hyp

    def random_error_correction(self):
        n = self.x.shape[0]
        index = np.random.choice(n * n, self.num_samples, replace=False)
        xu = self.x[self.inducing_points]

        Kuu = self.kernel.evaluate(xu, xu, self.hyp) + 1e-6 * np.eye(len(self.inducing_points))
        Luu = cholesky(Kuu, lower=True)
        Kuu_inv = cho_solve((Luu, True), np.eye(len(self.inducing_points)))
        for num in index:
            row = num / n
            col = num % n
            if row not in self.Omega:
                self.Omega[row] = []
                self.values[row] = []

            true_kernel = self.kernel.evaluate(self.x[row], self.x[col], self.hyp)
            approx_kernel = self.kernel.evaluate(self.x[row], xu, self.hyp)
            approx_kernel = np.dot(approx_kernel, Kuu_inv)
            approx_kernel = np.dot(approx_kernel, self.kernel.evaluate(xu, self.x[col], self.hyp))

            dif = true_kernel - approx_kernel
            self.Omega[row].append(col)
            self.values[row].append(dif[0][0])


def plot_k(k):
    plt.pcolor(np.abs(k))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.clim(0, 1e-6)
    plt.show()


def test():
    n = 500
    m = 50
    x = np.random.normal(0, 5, size=(n, 1))
    x = np.sort(x, axis=0)
    inducing = np.random.choice(n, m, replace=False)

    hyp = [np.log(5), np.log(5)]
    cov = Materniso(5)

    ka = KernelApproximate(cov, x, 5000, inducing, hyp)
    ka.random_error_correction()
    omg = ka.Omega
    val = ka.values

    mc = MatrixCompletion(n, 50, omg, val, 0.001)
    mc.completion(500)

    xu = x[inducing]
    true_k = cov.evaluate(x, x, hyp)

    Kuu = cov.evaluate(xu, xu, hyp)
    Luu = cholesky(Kuu + 1e-6 * np.eye(m), lower=True)
    Kuu_inv = cho_solve((Luu, True), np.eye(m))
    Kxu = cov.evaluate(x, xu, hyp)

    approx_error = np.dot(mc.U, mc.U.T)
    print np.linalg.norm(approx_error)

    approx_k = np.dot(Kxu, Kuu_inv)
    approx_k = np.dot(approx_k, Kxu.T)
    E = true_k - approx_k
    print np.linalg.norm(E)
    plot_k(E)

    E = true_k - (approx_k + approx_error)
    print np.linalg.norm(E)


if __name__ == '__main__':
    test()
