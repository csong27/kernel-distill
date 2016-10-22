from cov.se_iso import SEiso
from matrix_comp import MatrixCompletion
from scipy.linalg import cholesky, cho_solve
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123132)


class KernelApproximate(object):
    def __init__(self, kernel, x, numPoints, inducingPoints, hyp):
        self.kernel = kernel
        self.x = x
        self.numPoints = numPoints
        self.Omega = {}
        self.values = {}
        self.inducingPoints = inducingPoints
        self.hyp = hyp

    def random_error_correction(self):
        n = self.x.shape[0]
        index = np.random.choice(n * n, self.numPoints, replace=False)
        xu = self.x[self.inducingPoints, :]

        Kuu = self.kernel.evaluate(xu, xu, self.hyp) + 1e-6 * np.eye(len(self.inducingPoints))
        Luu = cholesky(Kuu, lower=True)
        Kuu_inv = cho_solve((Luu, True), np.eye(len(self.inducingPoints)))

        for num in index:
            row = num / n
            col = num % n
            if row not in self.Omega:
                self.Omega[row] = []
                self.values[row] = []

            true_kernel = self.kernel.evaluate(self.x[row], self.x[col], self.hyp)
            appro_kernel = self.kernel.evaluate(self.x[row], xu, self.hyp)
            appro_kernel = np.dot(appro_kernel, Kuu_inv)
            appro_kernel = np.dot(appro_kernel, self.kernel.evaluate(xu, self.x[col], self.hyp))

            dif = true_kernel - appro_kernel
            self.Omega[row].append(col)
            self.values[row].append(dif[0][0])


def test():
    x = np.random.normal(0, 5, size=(100, 1))
    x = np.sort(x, axis=0)
    inducing = np.arange(100)[::10]

    hyp = [np.log(1), np.log(0.1)]
    ka = KernelApproximate(SEiso, x, 1000, inducing, hyp)
    ka.random_error_correction()
    omg = ka.Omega
    val = ka.values

    mc = MatrixCompletion(100, 20, omg, val, 0.0001)
    mc.completion(1000)

    xu = x[inducing]
    se = SEiso()
    true_k = se.evaluate(x, x, hyp)

    Kuu = se.evaluate(xu, xu, hyp)
    Luu = cholesky(Kuu + 1e-6 * np.eye(len(inducing)), lower=True)
    Kuu_inv = cho_solve((Luu, True), np.eye(len(inducing)))
    Kxu = se.evaluate(x, xu, hyp)

    approx_error = np.dot(mc.U, mc.U.T)
    print np.linalg.norm(approx_error)

    approx_k = np.dot(Kxu, Kuu_inv)
    approx_k = np.dot(approx_k, Kxu.T)
    E = true_k - approx_k
    print np.linalg.norm(E)

    E = true_k - (approx_k + approx_error)
    print np.linalg.norm(E)


if __name__ == '__main__':
    test()
