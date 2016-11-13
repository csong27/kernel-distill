import numpy as np
from kernel import SEiso
from sklearn.decomposition import SparseCoder


def sp():
    n = 500
    dim = 1
    k = 70
    m = k ** dim
    X = np.random.normal(0, 5, size=(n, dim))
    X = np.sort(X, axis=0)
    inducing = np.random.choice(n, m, replace=False)
    inducing = np.sort(inducing)
    U = X[inducing]
    hyp = [np.log(20), np.log(2)]
    kernel = SEiso()
    K_xx = kernel.evaluate(X, X, hyp)
    K_uu = kernel.evaluate(U, U, hyp)
    K_xu = kernel.evaluate(X, U, hyp)
    coder = SparseCoder(dictionary=K_uu, transform_algorithm='lasso_lars', transform_alpha=1e-5, split_sign=True)
    W = coder.fit_transform(K_xu)
    K_approx = reduce(np.dot, [W, K_uu, W.T])
    print np.linalg.norm(K_xx - K_approx)

    indices = np.arange(len(W[0]))
    for i, row in enumerate(W):
        print i, indices[row > 0]


if __name__ == '__main__':
    sp()