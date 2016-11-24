from kernel_distill import Distillation
from kernel import GPMLKernel
from gp import GaussianProcess
from data_utils import load_kin40k, load_pumadyn32nm
import scipy.io as sio
import numpy as np
import cPickle as pkl
import os


def load_trained_hyp(dataset='pumadyn32nm'):
    fname = '../model/hyp_{}.mat'.format(dataset)
    hyp_mat = sio.loadmat(fname)['hyp']
    hyp_cov = hyp_mat['cov'][0][0].flatten()
    hyp_lik = hyp_mat['lik'][0][0][0][0]
    hyp = {'mean': [], 'lik': float(hyp_lik), 'cov': hyp_cov.astype(float)}
    return hyp


def experiment_kin40k():
    train_x, train_y, test_x, test_y = load_kin40k()
    pass


def experiment_pumadyn32nm(use_kmeans=True, m=100, reduce_from='fitc', cov='covSEard'):
    train_x, train_y, test_x, test_y = load_pumadyn32nm()
    n, d = train_x.shape
    print 'Training with {} data points and {} dimension'.format(n, d)
    # get GP functionality
    gp = GaussianProcess()
    # subtract mean
    train_y -= np.mean(train_y)
    test_y -= np.mean(train_y)
    # initialization
    hyp_cov = np.asarray([np.log(5)] * (d + 1))
    hyp_lik = float(np.std(train_y))
    hyp_old = {'mean': [], 'lik': hyp_lik, 'cov': hyp_cov}
    # train the kernel to reduce from
    # load hyp if trained already
    hyp_fname = '../model/pumadyn32nm_hyp_{}.pkl'.format(reduce_from)
    if os.path.exists(hyp_fname):
        print 'Loady hyperparams from {}'.format(hyp_fname)
        f = open(hyp_fname, 'rb')
        hyp = pkl.load(f)
        f.close()
    else:
        print 'Training the given kernel with {}'.format(reduce_from)
        if reduce_from == 'exact':
            f = open(hyp_fname, 'wb')
            hyp = gp.train_exact(train_x, train_y.reshape(-1, 1), hyp_old)
            pkl.dump(hyp, f, -1)
            f.close()
        elif reduce_from == 'fitc':
            xu = np.random.choice(n, 1024, replace=False)
            xu = train_x[xu]
            hyp = load_trained_hyp()
            test_mean, test_var = gp.predict_fitc(train_x, train_y.reshape(-1, 1),
                                                  xu=xu, xstar=test_x, hyp=hyp, cov=cov)
            mae = np.mean(np.abs(test_mean - test_y))   # / np.mean(np.abs(test_y - np.mean(test_y)))
            smse = np.mean(np.square(test_mean - test_y)) / np.var(test_y)
            print mae, smse
        else:
            raise ValueError(reduce_from)

    print hyp
    hyp_cov = hyp['cov'][0]
    sigmasq = np.exp(2 * hyp['lik'])
    kernel = GPMLKernel(cov)
    ind = np.random.choice(n, m, replace=False)
    U = train_x[ind]
    # distill the kernel
    distill = Distillation(X=train_x, y=train_y, U=U, kernel=kernel, hyp=hyp_cov, num_iters=100, eta=1e-2,
                           sigmasq=sigmasq, width=20, use_kmeans=use_kmeans, optimizer='sgd')
    distill.grad_descent()
    distill.precompute(use_true_K=False)


if __name__ == '__main__':
    experiment_pumadyn32nm(m=1000, use_kmeans=True)
    # load_trained_hyp()