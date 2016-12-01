import numpy as np
import matplotlib.pyplot as plt
from kernel_distill import Distillation
from kernel import SEiso
from gp import GaussianProcess


def experiment1D():
    gp = GaussianProcess()
    # setup data
    n = 500
    m = 50
    f = lambda x: np.sin(x) * np.exp(-x**2 / 50)
    X = np.random.uniform(-10, 10, size=n)
    X = np.sort(X)
    y = f(X) + np.random.normal(0, 1, size=n)
    y -= np.mean(y)

    x_min, x_max = np.min(X), np.max(X)
    U = np.linspace(x_min, x_max, m).reshape(-1, 1)

    X = X.reshape(-1, 1)
    hyp_cov = np.asarray([np.log(1), np.log(2)])
    hyp_lik = float(np.log(1))
    hyp_old = {'mean': [], 'lik': hyp_lik, 'cov': hyp_cov}
    hyp = gp.train_exact(X, y.reshape(-1, 1), hyp_old)
    hyp_cov = hyp['cov'][0]
    sigmasq = np.exp(2 * hyp['lik'])
    kernel = SEiso()
    distill = Distillation(X=X, y=y, U=U, kernel=kernel, hyp=hyp_cov, num_iters=10, eta=5e-4,
                           sigmasq=sigmasq, width=3, use_kmeans=True, optimizer='sgd')
    distill.grad_descent()
    distill.precompute(use_true_K=False)

    xx = np.linspace(x_min, x_max, 2 * n)
    mm_true, vv_true = gp.predict_exact(X, y.reshape(-1, 1), xx.reshape(-1, 1), hyp=hyp)

    mm = []
    vv = []

    opt = {'cg_maxit': 500, 'cg_tol': 1e-5}
    k = n / 2
    hyp = gp.train_kiss(X, y.reshape(-1, 1), k, hyp=hyp_old, opt=opt)
    mm_kiss, vv_kiss = gp.predict_kiss(X, y.reshape(-1, 1), xx.reshape(-1, 1), k, hyp=hyp, opt=opt)

    for xstar in xx:
        xstar = np.asarray([xstar])
        mstar, vstar = distill.predict(xstar, width=3)
        vv.append(vstar)
        mm.append(mstar)

    mm = np.asarray(mm).flatten()
    vv = np.asarray(vv).flatten()
    mm_kiss = np.asarray(mm_kiss).flatten()
    vv_kiss = np.asarray(vv_kiss).flatten()

    plt.fill_between(xx, mm - np.sqrt(vv) * 2, mm + np.sqrt(vv) * 2, color='gray', alpha=.5)
    plt.plot(xx, mm_true, color='y', lw=3, label='exact mean')
    plt.plot(xx, mm, color='r', lw=3, label='distill mean', ls='dotted')
    plt.plot(xx, mm_kiss, color='g', lw=3, label='kiss mean', ls=':')
    plt.plot(xx, f(xx), lw=3, label='true value', ls='dashed')
    plt.scatter(X, y, color='m', label='train data', marker='+')
    plt.xlim([x_min, x_max])
    plt.legend()
    plt.show()

    plt.plot(xx, vv_kiss, color='g', lw=3, label='kiss var', ls=':')
    plt.plot(xx, vv_true, color='y', lw=3, label='exact var')
    plt.plot(xx, vv, color='r', lw=3, label='distill var', ls='dotted')
    plt.xlim([x_min, x_max])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    experiment1D()
