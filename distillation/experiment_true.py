from kernel_distill import Distillation
from kernel import SEiso
from gp import GaussianProcess
from data_utils import load_kin40k, load_pumadyn32nm
import numpy as np


def experiment_kin40k():
    train_x, train_y, test_x, test_y = load_kin40k()
    pass


def experiment_pumadyn32nm(use_kmeans=True, m=1024):
    train_x, train_y, test_x, test_y = load_pumadyn32nm()
    n = len(train_x)

    gp = GaussianProcess()

    train_y -= np.mean(train_y)
    hyp_cov = np.asarray([np.log(1), np.log(2)])
    hyp_lik = float(np.log(1))
    hyp_old = {'mean': [], 'lik': hyp_lik, 'cov': hyp_cov}
    hyp = gp.train_exact(train_x, train_y.reshape(-1, 1), hyp_old)
    hyp_cov = hyp['cov'][0]
    sigmasq = np.exp(2 * hyp['lik'])
    kernel = SEiso()
    if not use_kmeans:
        U = np.random.choice(n, m, replace=False)
    else:
        U = None

    distill = Distillation(X=train_x, y=train_y, U=U, kernel=kernel, hyp=hyp_cov, num_iters=10, eta=5e-4,
                           sigmasq=sigmasq, width=10, use_kmeans=use_kmeans, optimizer='sgd')
    distill.grad_descent()
    distill.precompute(use_true_K=False)