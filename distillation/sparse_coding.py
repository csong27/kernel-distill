import numpy as np


def q3():
    A = [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [1, 1, 0, 0],
         [1, 1, 0, 0]]

    L = 2 * np.eye(4) - A
    L = np.asarray(L)
    v, w = np.linalg.eig(L)
    print w[:, 2]
    print np.dot(L, w[:, 2])


def q2(k):
    beta1 = (k - 1) / (1.01 * k)
    beta2 = 1 / 1.01
    f = lambda beta: np.exp(-0.505*k * (beta* np.log(beta) - beta + 1))
    print 0.5 * (f(beta1) + f(beta2))


if __name__ == '__main__':
    q2(120527)
