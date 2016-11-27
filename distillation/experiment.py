from kernel_distill import Distillation
from eval_metrics import smse, smae
from kernel import SEard, SEiso
from gp import GaussianProcess
from data_utils import *
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import numpy as np
import scipy.linalg as sl
import cPickle as pkl
import os

np.random.seed(21345)


def get_fitc_kernel(x, xu, hyp, kernel):
    m = xu.shape[0]
    K_xu = kernel.evaluate(x, xu, hyp)
    K_uu = kernel.evaluate(xu, xu, hyp)
    K_xx = kernel.evaluate(x, x, hyp)
    L_uu = sl.cholesky(K_uu + 1e-6 * np.eye(m), lower=True)
    K_uu_inv = sl.cho_solve((L_uu, True), np.eye(m))
    K_sor = reduce(np.dot, [K_xu, K_uu_inv, K_xu.T])
    K_diag = np.diag(K_xx - K_sor)
    K_fitc = K_sor + np.diag(K_diag)
    return K_fitc


def load_trained_hyp(dataset=PUMADYN32NM):
    fname = '../model/hyp_{}.mat'.format(dataset)
    hyp_mat = sio.loadmat(fname)['hyp']
    hyp_cov = hyp_mat['cov'][0][0].flatten()
    hyp_lik = hyp_mat['lik'][0][0][0][0]
    hyp = {'mean': [], 'lik': float(hyp_lik), 'cov': hyp_cov.astype(float)}
    return hyp


def experiment(dataset=PUMADYN32NM, use_kmeans=True, m=100, reduce_from='fitc', cov='covSEard', width=20,
               standardize=True, load_trained=False):
    train_x, train_y, test_x, test_y = load_dataset(dataset)
    if standardize:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    n, d = train_x.shape
    print 'Distilling with {} data points and {} dimension'.format(n, d)
    # get GP functionality
    gp = GaussianProcess()
    # subtract mean
    train_y -= np.mean(train_y)
    test_y -= np.mean(train_y)
    # initialization
    hyp_lik = float(0.5 * np.log(np.var(train_y) / 4))
    if cov == 'covSEard':
        init_ell = np.log((np.max(train_x, axis=0) - np.min(train_x, axis=0)) / 2)
        hyp_cov = np.append(init_ell, [0.5 * np.log(np.var(train_y))])
    else:
        hyp_cov = np.asarray([np.log(2)] * 2)

    hyp_old = {'mean': [], 'lik': hyp_lik, 'cov': hyp_cov}
    # train the kernel to reduce from
    # load hyp if trained already
    xu = np.random.choice(n, m, replace=False)
    xu = train_x[xu]
    hyp_fname = '../model/{}_hyp_{}.pkl'.format(dataset, reduce_from)

    if load_trained and os.path.exists(hyp_fname):
        print 'Loady hyperparams from {}'.format(hyp_fname)
        f = open(hyp_fname, 'rb')
        hyp = pkl.load(f)
        f.close()
    else:
        print 'Training the given kernel with {}'.format(reduce_from)
        if reduce_from == 'fitc':
            if dataset in {PUMADYN32NM, KIN40K}:
                hyp = load_trained_hyp(dataset)
            else:
                hyp = gp.train_fitc(train_x, train_y.reshape(-1, 1), xu, hyp_old, cov=cov)
            test_mean, test_var = gp.predict_fitc(train_x, train_y.reshape(-1, 1),
                                                  xu=xu, xstar=test_x, hyp=hyp, cov=cov)
            f = open(hyp_fname, 'wb')
            pkl.dump(hyp, f, -1)
            f.close()
        elif reduce_from == 'exact':
            hyp = gp.train_exact(train_x, train_y.reshape(-1, 1), hyp_old, cov=cov, n_iter=100)
            test_mean, test_var = gp.predict_exact(train_x, train_y.reshape(-1, 1), xstar=test_x, hyp=hyp, cov=cov)
            f = open(hyp_fname, 'wb')
            pkl.dump(hyp, f, -1)
            f.close()
        else:
            raise ValueError(reduce_from)

        print smae(test_y, test_mean), smse(test_y, test_mean)

    hyp_cov = hyp['cov'].flatten()
    sigmasq = np.exp(2 * hyp['lik'])
    kernel = SEard() if cov == 'covSEard' else SEiso()

    # distill the kernel
    distill = Distillation(X=train_x, y=train_y, U=xu, kernel=kernel, hyp=hyp_cov, num_iters=0, eta=1e-2,
                           sigmasq=sigmasq, width=width, use_kmeans=use_kmeans, optimizer='adagrad')
    distill.grad_descent()
    distill.precompute(use_true_K=False)

    # plt.pcolor(np.abs(distill.diff_to_K()[:2000, :2000]))
    # plt.colorbar()
    # plt.show()

    mm, vv = [], []
    for xstar in test_x:
        xstar = xstar.reshape(1, d)
        mstar, vstar = distill.predict(xstar, width=width)
        vv.append(vstar)
        mm.append(mstar)

    mm = np.asarray(mm).flatten()

    print 'Distill Error:'
    print_error(test_y, mm)
    print 'Mean error to true K:'
    print_error(test_mean, mm)


def experiment_kiss(dataset=PUMADYN32NM, m=100, proj_d=2, cov='covSEard', standardize=True, proj=None):
    assert proj in {None, 'norm', 'orth'}
    train_x, train_y, test_x, test_y = load_dataset(dataset)
    if standardize:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    n, d = train_x.shape
    print 'Distilling with {} data points and {} dimension'.format(n, d)
    # get GP functionality
    gp = GaussianProcess()
    # subtract mean
    train_y -= np.mean(train_y)
    test_y -= np.mean(train_y)
    # projection matrix
    P = np.random.normal(size=(proj_d, d))
    # initialization
    hyp_lik = float(0.5 * np.log(np.var(train_y) / 4))
    if cov == 'covSEard':
        init_x = np.dot(train_x, P.T)
        init_ell = np.log((np.max(init_x, axis=0) - np.min(init_x, axis=0)) / 2)
        hyp_cov = np.append(init_ell, [0.5 * np.log(np.var(train_y))])
    else:
        hyp_cov = np.asarray([np.log(2)] * 2)

    hyp_old = {'mean': [], 'lik': hyp_lik, 'cov': hyp_cov, 'proj': P}
    opt = {'cg_maxit': 500, 'cg_tol': 1e-5}
    if proj is not None:
        opt['proj'] = proj
    hyp = gp.train_kiss(train_x, train_y.reshape(-1, 1), k=m, hyp=hyp_old, opt=opt, n_iter=100)
    test_mean, test_var = gp.predict_kiss(train_x, train_y.reshape(-1, 1), k=m, xstar=test_x, opt=opt, hyp=hyp, cov=cov)

    print 'Fast KISS error:'
    print_error(test_y, test_mean)
    test_mean, test_var = gp.predict_kiss(train_x, train_y.reshape(-1, 1), fast=False,
                                          k=m, xstar=test_x, opt=opt, hyp=hyp, cov=cov)
    print 'Slow KISS error:'
    print_error(test_y, test_mean)


def print_error(target, predictions):
    print "SMAE: {}, SMSE: {}".format(smae(target, predictions), smse(target, predictions))


if __name__ == '__main__':
    experiment_kiss(dataset=BOSTON, m=100, cov='covSEiso', proj='orth', proj_d=2)
