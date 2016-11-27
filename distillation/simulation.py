import numpy as np
import matplotlib.pyplot as plt
from kernel_distill import Distillation
from kernel import SEiso
from gp import GaussianProcess


def experiment1D():
    gp = GaussianProcess()
    # setup data
    n = 1000
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
                           sigmasq=sigmasq, width=10, use_kmeans=True, optimizer='sgd')
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
        vv.append(distill.predict_variance(xstar, width=10))
        mm.append(distill.predict_mean(xstar, width=10))

    mm = np.asarray(mm).flatten()
    vv = np.asarray(vv).flatten()
    mm_kiss = np.asarray(mm_kiss).flatten()
    vv_kiss = np.asarray(vv_kiss).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.fill_between(xx, mm - np.sqrt(vv) * 2, mm + np.sqrt(vv) * 2, color='gray', alpha=.5)
    ax1.plot(xx, mm_true, color='y', lw=3, label='exact mean')
    ax1.plot(xx, mm, color='r', lw=3, label='distill mean', ls='dotted')
    ax1.plot(xx, mm_kiss, color='m', lw=3, label='kiss mean', ls=':')
    ax1.plot(xx, f(xx), lw=3, label='true value', ls='dashed')
    ax1.scatter(X, y, color='g', label='train data', marker='+')
    ax1.set_xlim([x_min, x_max])
    ax1.legend()

    ax2.plot(xx, vv_kiss, color='m', lw=2, label='kiss var', ls=':')
    ax2.plot(xx, vv_true, color='y', lw=2, label='exact var')
    ax2.plot(xx, vv, color='r', lw=2, label='distill var', ls='dotted')
    ax2.set_xlim([x_min, x_max])
    ax2.legend()
    plt.show()


def experiment2D():
    gp = GaussianProcess()
    # setup data
    n = 100
    m = 30
    f = lambda x: np.sin(x[:, 1]) + x[:, 0]
    X = np.random.uniform(-3, 3, size=(n, 2))
    y = f(X) + np.random.normal(0, 0.1, size=n)
    mean_y = np.mean(y)
    y -= mean_y
    U = X[np.random.choice(n, m, replace=False)]

    # setup test data
    x1, x2 = np.mgrid[-4: 4: 100j, -4: 4: 100j]
    xx = np.vstack([x1.ravel(), x2.ravel()])
    # print xx.shape
    xx = xx.T
    yy = f(xx)
    yy -= mean_y

    hyp_cov = np.asarray([np.log(2), np.log(2)])
    hyp_lik = float(np.log(2))
    hyp_old = {'mean': [], 'lik': hyp_lik, 'cov': hyp_cov}
    print 'Training exact'
    hyp = gp.train_exact(X, y.reshape(-1, 1), hyp_old, n_iter=100)
    mm_true, vv_true = gp.predict_exact(X, y.reshape(-1, 1), xx, hyp=hyp)
    print '--------------------'
    print hyp
    hyp_cov = hyp['cov'][0]
    sigmasq = np.exp(2 * hyp['lik'])
    kernel = SEiso()
    distill = Distillation(X=X, y=y, U=U, kernel=kernel, hyp=hyp_cov, num_iters=100, eta=2e-6,
                           sigmasq=sigmasq, width=10, use_kmeans=True, optimizer='sgd')
    distill.grad_descent()
    distill.precompute(use_true_K=False)

    mm = []
    vv = []
    opt = {'cg_maxit': 500, 'cg_tol': 1e-5}
    k = m * 2
    hyp = gp.train_kiss(X, y.reshape(-1, 1), k, hyp=hyp_old, opt=opt, n_iter=1)
    mm_kiss, vv_kiss = gp.predict_kiss(X, y.reshape(-1, 1), xx, k, hyp=hyp, opt=opt)

    for xstar in xx:
        xstar = xstar.reshape(-1, 2)
        vv.append(distill.predict_variance(xstar, width=10))
        mm.append(distill.predict_mean(xstar, width=10))

    mm = np.asarray(mm).flatten()
    vv = np.asarray(vv).flatten()
    mm_kiss = np.asarray(mm_kiss).flatten()
    vv_kiss = np.asarray(vv_kiss).flatten()

    # plt.imshow(mm.reshape(100, 100))
    # plt.show()
    #
    # plt.imshow(mm_kiss.reshape(100, 100))
    # plt.show()

    f_mae_distill = np.abs(mm - yy)     # / np.abs(yy - np.mean(yy))
    print np.mean(f_mae_distill)

    f_mae_kiss = np.abs(mm_kiss - yy) # / np.abs(yy - np.mean(yy))
    print np.mean(f_mae_kiss)

    mean_mae_distill = np.abs(mm - mm_true) # / np.abs(mm_true - np.mean(mm_true))
    print np.mean(mean_mae_distill)

    var_mae_distill = np.abs(vv - vv_true) # / np.abs(vv_true - np.mean(vv_true))
    print np.mean(var_mae_distill)

    mean_mae_kiss = np.abs(mm_kiss - mm_true) # / np.abs(mm_true - np.mean(mm_true))
    print np.mean(mean_mae_kiss)

    var_mae_kiss = np.abs(vv_kiss - vv_true) # / np.abs(vv_true - np.mean(vv_true))
    print np.mean(var_mae_kiss)

if __name__ == '__main__':
    experiment2D()
