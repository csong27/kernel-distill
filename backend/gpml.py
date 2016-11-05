"""
GPML backend for Gaussian processes.
"""
import os
import numpy as np
from backend.engines import MATLABEngine as Engine
from pprint import pprint

GPML_PATH = 'D:\Course\Cornell\orie6741\gpml-matlab-v4.0-2016-10-19/'

# MATLAB scripts
_gp_train_epoch = """
hyp = minimize(hyp, @gp, -{n_iter:d}, {inf}, {mean}, {cov}, {lik}, X_tr, y_tr);
"""
_gp_predict = """
[ymu ys2 fmu fs2   ] = gp(hyp, {inf}, {mean}, {cov}, {lik}, X_tr, y_tr, X_tst);
"""
_gp_evaluate = """
[nlZ dnlZ          ] = gp(hyp, {inf}, {mean}, {cov}, {lik}, {X}, {y});
"""
_gp_dlik = """
[dlik_dx           ] = dlik(hyp, {mean}, {cov}, {lik}, {dcov}, {X}, {y});
"""
_gp_create_grid = """
xg = apxGrid('create', {X}, eq, k);
"""
_gp_grid_interpolate = """
[row col val dval N] = apxGridUtils('interp', xg, {X}, deg);
"""

_gp_dcov_evaluation = """
dK = feval(dK, {Q});
"""

_gp_cov_evaluation = """
[K dK] = feval({cov}, hyp, {X});
"""

_gp_structure_grid_K = """
[Kg Mx] = apxGrid({cov}, {xg}, {hyp}, {X});
"""

_gp_expand_grid = """
[xe nx Dx] = apxGrid('expand', xg);
"""


class GPML(object):
    """Class that implements backend functionality for Gaussian processes.
    Arguments:
    ----------
        engine : str (either 'matlab' or 'octave')
        gpml_path : str or None
    """
    def __init__(self, engine='matlab', engine_kwargs={}, gpml_path=GPML_PATH):
        if engine is None:
            if 'GP_ENGINE' in os.environ:
                engine = os.environ['GP_ENGINE']
            else:
                raise ValueError("GP_ENGINE is neither provided nor available "
                                 "in the environment.")
        if engine != 'matlab':
            raise ValueError('Unknown GP_ENGINE: %s' % engine)

        if gpml_path is None:
            if 'GPML_PATH' in os.environ:
                gpml_path = os.environ['GPML_PATH']
            else:
                current_dir = os.path.dirname(os.path.realpath(__file__))
                gpml_path = os.path.join(current_dir, 'backend')
                if not os.path.isfile(os.path.join(gpml_path, 'startup.m')):
                    raise ValueError(
                        "Neither GPML_PATH is provided nor GPML library is "
                        "available directly from KGP. "
                        "Please make sure you cloned KGP *recursively*.")

        self.eng = Engine(**engine_kwargs)
        self.eng.addpath(gpml_path)
        self.eng.eval('startup', verbose=0)

        utils_path = os.path.join(os.path.dirname(__file__), 'gpml_utils')
        self.eng.addpath(utils_path)

    def configure(self, input_dim, hyp, opt, inf, mean, cov, lik, dlik,
                  grid_kwargs=None, verbose=1):
        """Configure GPML-based Guassian process.
        Arguments:
        ----------
            input_dim : uint
                The dimension of the GP inputs.
            hyp : dict
                A dictionary of GP hyperparameters.
            opt : dict
                GPML inference/training options (see GPML doc for details).
            inf : str
                Name of the inference method (see GPML doc for details).
            mean : str
                Name of the mean function (see GPML doc for details).
            cov : str
                Name of the covariance function (see GPML doc for details).
            lik : str
                Name of the likelihood function (see GPML doc for details).
            dlik : str
                Name of the function that computes dlik/dx.
            grid_kwargs : dict
                'eq' : uint (default: 1)
                    Whether to enforce an equispaced grid.
                'k' : uint or float in (0, 1]
                    Number of inducing points per dimension.
        """
        self.config = {}
        self.config['lik'] = "{@%s}" % lik
        self.config['mean'] = "{@%s}" % mean
        self.config['inf'] = "{@(varargin) %s(varargin{:}, opt)}" % inf
        self.config['dlik'] = "@(varargin) %s(varargin{:}, opt)" % dlik

        if inf == 'infGrid':
            assert grid_kwargs is not None, \
                "GPML: No arguments provided for grid generation for infGrid."
            self.eng.push('k', grid_kwargs['k'])
            self.eng.push('eq', grid_kwargs['eq'])
            cov = ','.join(input_dim * ['{@%s}' % cov])
            if input_dim > 1:
                cov = '{' + cov + '}'
            hyp['cov'] = np.tile(hyp['cov'], (1, input_dim))
            self.config['cov'] = "{@covGrid, %s, xg}" % cov
            self.config['dcov'] = "[]"
        else:
            hyp['cov'] = np.asarray(hyp['cov'])
            self.config['cov'] = "{@%s}" % cov
            self.config['dcov'] = "@d%s" % cov

        self.eng.push('hyp', hyp)
        self.eng.push('opt', opt)
        self.eng.eval("dlik = %s;" % self.config['dlik'], verbose=0)

        if verbose:
            print("GP configuration:")
            pprint(self.config)

    def update_data(self, which_set, X, y=None):
        """Update data in GP backend.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        self.eng.push('X_' + which_set, X)
        if y is not None:
            self.eng.push('y_' + which_set, y)

    def update_grid(self, which_set):
        """Update grid for grid-based GP inference.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        self.config.update({'X': 'X_' + which_set, 'y': None})
        self.eng.eval(_gp_create_grid.format(**self.config), verbose=0)

    def evaluate(self, which_set, X=None, y=None, verbose=0):
        """Evaluate GP for given X and y.
        Return negative log marginal likelihood.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        if X is not None and y is not None:
            self.update_data(which_set, X, y)
        X_name, y_name = 'X_' + which_set, 'y_' + which_set
        self.config.update({'X': X_name, 'y': y_name})
        self.eng.eval(_gp_evaluate.format(**self.config), verbose=verbose)
        nlZ = self.eng.pull('nlZ')
        return nlZ

    def predict(self, X, X_tr=None, y_tr=None, return_var=False, verbose=0):
        """Predict ymu and ys2 for a given X. Return ymu and ys2.
        """
        self.update_data('tst', X)
        if X_tr is not None and y_tr is not None:
            self.update_data('tr', X_tr, y_tr)
        self.eng.eval(_gp_predict.format(**self.config), verbose=verbose)
        preds = self.eng.pull('ymu')
        if return_var:
            preds = (preds, self.eng.pull('ys2'))
        return preds

    def train(self, n_iter, X_tr=None, y_tr=None, verbose=0):
        """Train GP for `n_iter` iterations. Return a dict of hyperparameters.
        """
        if X_tr is not None and y_tr is not None:
            self.update_data('tr', X_tr, y_tr)
        self.config.update({'n_iter': n_iter})
        self.eng.eval(_gp_train_epoch.format(**self.config), verbose=verbose)
        hyp = self.eng.pull('hyp')
        return hyp

    def get_dlik_dx(self, which_set, verbose=0):
        """Get derivative of the log marginal likelihood w.r.t. the kernel.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        X_name, y_name = 'X_' + which_set, 'y_' + which_set
        self.config.update({'X': X_name, 'y': y_name})
        self.eng.eval(_gp_dlik.format(**self.config), verbose=verbose)
        dlik_dx = self.eng.pull('dlik_dx')
        return dlik_dx

    def create_grid(self, X, eq, k, rtn_py=False):
        if isinstance(k, int):
            k = float(k)
        config = {'X': 'X'}
        self.eng.push('X', X)
        self.eng.push('eq', eq)
        self.eng.push('k', k)
        self.eng.eval(_gp_create_grid.format(**config), verbose=0)
        xg = self.eng.pull('xg')
        if rtn_py:
            return np.asarray(xg).reshape(-1, k)
        return xg

    def interpolate_grid(self, X, eq, k, deg=3., expand_grid=False):
        config = {'X': 'X'}
        xg = self.create_grid(X, eq, k, False)
        self.eng.push('X', X)
        self.eng.push('xg', xg)
        self.eng.push('deg', deg)
        self.eng.eval(_gp_grid_interpolate.format(**config), verbose=1)
        row = self.eng.pull('row').flatten() - 1    # -1 due to matlab index
        col = self.eng.pull('col').flatten() - 1    # -1 due to matlab index
        val = self.eng.pull('val').flatten()
        dval = self.eng.pull('dval')
        N = self.eng.pull('N')
        if expand_grid:
            xg = self.expand_grid(xg)

        return xg, row, col, val, dval, N

    def cov_eval(self, X, cov, hyp, Q=None):
        config = {'X': 'X', 'cov': "@%s" % cov}
        hyp = np.asarray(hyp)
        self.eng.push('X', X)
        self.eng.push('hyp', hyp)
        self.eng.eval(_gp_cov_evaluation.format(**config), verbose=1)
        K = self.eng.pull('K')
        if Q is not None:
            config = {'Q': 'Q'}
            self.eng.push('Q', Q)
            self.eng.eval(_gp_dcov_evaluation.format(**config), verbose=1)
            dK = self.eng.pull('dK')
            return K, dK
        return K

    def struct_K_mvm(self, xg, hyp, cov, X):
        input_dim = X.shape[1]
        cov = ','.join(input_dim * ['{@%s}' % cov])
        if input_dim > 1:
            cov = '{' + cov + '}'
        hyp = np.tile(hyp, (1, input_dim))
        config = {'X': 'X', 'cov': cov, 'hyp': 'hyp', 'xg': 'xg'}
        self.eng.push('X', X)
        self.eng.push('xg', xg)
        self.eng.push('hyp', hyp)
        self.eng.eval(_gp_structure_grid_K.format(**config), verbose=1)
        # self.eng.push('y', y)
        # config = {'y': 'y'}
        # self.eng.eval(_gp_structure_mvm.format(**config))
        # return self.eng.pull('y')

    def struct_K_der(self, Kg, xg, Mx, a, b):
        pass

    def expand_grid(self, xg):
        self.eng.push('xg', xg)
        self.eng.eval(_gp_expand_grid, verbose=1)
        xe = self.eng.pull('xe')
        return xe


def test():
    gpml = GPML()
    n = 20
    x = np.random.normal(5, size=(n, 2))
    hyp = [np.log(2), np.log(2)]
    cov = 'covSEiso'


if __name__ == '__main__':
    test()