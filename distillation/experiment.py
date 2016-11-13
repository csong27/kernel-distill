import numpy as np
import matplotlib.pyplot as plt
from kiss_distill import Distillation
from kernel import SEiso
from gp import GaussianProcess


def experiment1():
    gp = GaussianProcess()
    n = 100
    m = 15
    f = lambda x: 0.3 * x + 1.2 + np.sin(x)
    epsilon = np.random.normal(0, 1, size=n)
    X = 2 * np.random.normal(size=n) - 1
    X = 1 + 4 * X + np.sign(X)
    X = np.sort(X)
    y = f(X) + epsilon
    mean_y = np.mean(y)
    y -= mean_y
    X = X.reshape(-1, 1)
    inducing = np.random.choice(n, m, replace=False)
    inducing = np.sort(inducing)
    U = X[inducing]
    x_min, x_max = np.min(X), np.max(X)
    U = np.linspace(x_min, x_max, m).reshape(-1, 1)

    hyp_cov = np.asarray([np.log(1), np.log(2)])
    hyp_lik = float(np.log(1))
    hyp_old = {'mean': [], 'lik': hyp_lik, 'cov': hyp_cov}
    hyp = gp.train_exact(X, y.reshape(-1, 1), hyp_old)
    print hyp
    hyp_cov = hyp['cov'][0]
    sigmasq = 2 * np.exp(hyp['lik'])
    kernel = SEiso()
    distill = Distillation(X=X, y=y, U=U, kernel=kernel, hyp=hyp_cov, num_iters=1, eta=1e-3,
                           sigmasq=sigmasq, width=10)
    distill.grad_descent()

    mm = []
    vv = []
    x_min, x_max = np.min(X), np.max(X)
    xx = np.linspace(x_min, x_max, 100)

    opt = {'cg_maxit': 500, 'cg_tol': 1e-5}
    hyp = gp.train_kiss(X, y.reshape(-1, 1), 50, hyp=hyp_old, opt=opt)
    mm_true, vv_true = gp.predict_kiss(X, y.reshape(-1, 1), xx.reshape(-1, 1), m, hyp=hyp, opt=opt)

    # xx = X.flatten()
    for xstar in xx:
        xstar = np.asarray([xstar])
        vv.append(distill.predict_variance(xstar, width=10))
        mm.append(distill.predict_mean(xstar, width=10))

    mm = np.asarray(mm).flatten()
    vv = np.asarray(vv).flatten()
    mm_true = np.asarray(mm_true).flatten()
    vv_true = np.asarray(vv_true).flatten()

    plt.fill_between(xx, mm - np.sqrt(vv) * 2, mm + np.sqrt(vv) * 2, color='gray', alpha=.5)
    plt.plot(xx, mm, color='g', lw=1.5)
    plt.plot(xx, f(xx), lw=1.5)
    plt.scatter(X, y)
    plt.xlim(x_min, x_max)
    plt.show()

    plt.fill_between(xx, mm_true - np.sqrt(vv_true) * 2, mm_true + np.sqrt(vv_true) * 2, color='gray', alpha=.5)
    plt.plot(xx, mm_true, color='r', lw=1.5)
    plt.plot(xx, f(xx), lw=1.5)
    plt.scatter(X, y)
    plt.xlim(x_min, x_max)
    plt.show()


def experiment2():
    gp = GaussianProcess()
    # setup data
    n = 1000
    m = 50
    f = lambda x: np.sin(x) * np.exp(-x**2 / 50)
    X = np.random.uniform(-10, 10, size=n)
    X = np.sort(X)
    y = f(X) + np.random.normal(0, 1, size=n)
    y -= np.mean(y)

    plt.plot(X, y)
    plt.show()

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
    distill = Distillation(X=X, y=y, U=U, kernel=kernel, hyp=hyp_cov, num_iters=100, eta=1e-4,
                           sigmasq=sigmasq, width=10, use_kmeans=True)
    distill.grad_descent()
    distill.precompute(False)

    mm = []
    vv = []
    xx = np.linspace(x_min, x_max, 2 * n)

    opt = {'cg_maxit': 500, 'cg_tol': 1e-5}
    k = n / 2
    hyp = gp.train_kiss(X, y.reshape(-1, 1), k, hyp=hyp_old, opt=opt)
    mm_true, vv_true = gp.predict_kiss(X, y.reshape(-1, 1), xx.reshape(-1, 1), k, hyp=hyp, opt=opt)

    # xx = X.flatten()
    for xstar in xx:
        xstar = np.asarray([xstar])
        vv.append(distill.predict_variance(xstar, width=10))
        mm.append(distill.predict_mean(xstar, width=10))

    mm = np.asarray(mm).flatten()
    vv = np.asarray(vv).flatten()
    mm_true = np.asarray(mm_true).flatten()
    vv_true = np.asarray(vv_true).flatten()

    plt.fill_between(xx, mm - np.sqrt(vv) * 2, mm + np.sqrt(vv) * 2, color='gray', alpha=.5)
    plt.plot(xx, mm, color='g', lw=1.5)
    plt.plot(xx, f(xx), lw=1.5)
    plt.scatter(X, y)
    plt.xlim(x_min, x_max)
    plt.show()

    plt.fill_between(xx, mm_true - np.sqrt(vv_true) * 2, mm_true + np.sqrt(vv_true) * 2, color='gray', alpha=.5)
    plt.plot(xx, mm_true, color='r', lw=1.5)
    plt.plot(xx, f(xx), lw=1.5)
    plt.scatter(X, y)
    plt.xlim(x_min, x_max)
    plt.show()

if __name__ == '__main__':
    experiment2()
