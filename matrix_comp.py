import numpy as np


class MatrixCompletion(object):
    def __init__(self, n, r, Omega, values, eta):
        self.n = n
        self.r = r
        self.Omega = Omega
        self.values = values
        arr = np.concatenate(values.values())
        self.B = np.sqrt(np.max(arr))
        self.eta = eta
        self.U = np.sqrt(np.ones((n, r)) * abs(np.mean(arr)))

    def completion(self, num_simu):
        for i in xrange(num_simu):
            for row_ind in xrange(self.n):
                if row_ind in self.Omega:
                    self.update_row(row_ind)

    def update_row(self, row_ind):
        for j in xrange(len(self.Omega[row_ind])):
            col_ind = self.Omega[row_ind][j]
            grad = np.dot(self.U[row_ind], self.U[col_ind]) - self.values[row_ind][j]
            if col_ind == row_ind:
                grad = 2 * grad * self.U[col_ind]
            else:
                grad = grad * self.U[col_ind]
            self.U[row_ind] -= self.eta * grad

            norm = np.linalg.norm(self.U[row_ind])
            if norm > self.B:
                self.U[row_ind] = self.U[row_ind] / norm * self.B


if __name__ == '__main__':
    U = np.random.uniform(0, 1, size=(10, 2))
    M = np.dot(U, U.T)
    n = 10
    values = {}
    omg = {}
    for i in np.arange(n):
        omg[i] = np.sort(np.random.choice(n, 2, replace=False))
        values[i] = M[i][omg[i]]
    print omg

    mc = MatrixCompletion(n, 2, omg, values, 0.01)
    mc.completion(1000)
    print np.dot(mc.U,  mc.U.T) - M
    print "===================="
    print M
