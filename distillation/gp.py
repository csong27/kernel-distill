from backend.gpml import GPML


class GaussianProcess(object):
    def __init__(self):
        self.gpml = GPML()

    def train_exact(self, x, y, hyp, n_iter=100, mean='meanZero', cov='covSEiso'):
        inf = 'infGaussLik'
        input_dim = x.shape[1]
        opt = {}
        self.gpml.configure(input_dim=input_dim, hyp=hyp, opt=opt, mean=mean, cov=cov, inf=inf)
        self.gpml.update_data('tr', x, y)
        self.gpml.train(n_iter=n_iter, verbose=1)
        hyp = self.gpml.eng.pull('hyp')
        return hyp

    def predict_exact(self, x, y, xstar, hyp, mean='meanZero', cov='covSEiso'):
        inf = 'infGaussLik'
        input_dim = x.shape[1]
        opt = {}
        self.gpml.configure(input_dim=input_dim, hyp=hyp, opt=opt, mean=mean, cov=cov, inf=inf, verbose=0)
        return self.gpml.predict(xstar, x, y, return_var=True, verbose=1)

    def train_fitc(self, x, y, xu, hyp, n_iter=100, mean='meanZero', cov='covSEiso'):
        inf = 'infGaussLik'
        input_dim = x.shape[1]
        opt = {}
        cov = 'apxSparse,{@%s},xu' % cov
        self.gpml.configure(input_dim=input_dim, hyp=hyp, opt=opt, mean=mean, cov=cov, inf=inf)
        self.gpml.eng.push('xu', xu)
        self.gpml.update_data('tr', x, y)
        self.gpml.train(n_iter=n_iter, verbose=1)
        hyp = self.gpml.eng.pull('hyp')
        return hyp

    def predict_fitc(self, x, y, xu, xstar, hyp, mean='meanZero', cov='covSEiso'):
        inf = 'infGaussLik'
        input_dim = x.shape[1]
        opt = {}
        cov = 'apxSparse,{@%s},xu' % cov
        self.gpml.configure(input_dim=input_dim, hyp=hyp, opt=opt, mean=mean, cov=cov, inf=inf)
        self.gpml.eng.push('xu', xu)
        self.gpml.update_data('tr', x, y)
        return self.gpml.predict(xstar, return_var=True, verbose=1)

    def train_kiss(self, x, y, k, hyp, opt, n_iter=100, mean='meanZero', cov='covSEiso'):
        k = float(k)
        inf = 'infGrid'
        input_dim = x.shape[1]
        grid_kwargs = {'eq': 1, 'k': k}
        self.gpml.configure(input_dim=input_dim, hyp=hyp, opt=opt, mean=mean, cov=cov, inf=inf, grid_kwargs=grid_kwargs)
        self.gpml.update_data('tr', x, y)
        self.gpml.update_grid('tr')
        self.gpml.train(n_iter=n_iter, verbose=1)
        hyp = self.gpml.eng.pull('hyp')
        return hyp

    def predict_kiss(self, x, y, xstar, k, hyp, opt, mean='meanZero', cov='covSEiso', fast=True):
        inf = 'infGrid'
        input_dim = x.shape[1]
        grid_kwargs = {'eq': 1, 'k': float(k)}
        if fast:
            opt['pred_var'] = 20.0
        self.gpml.configure(input_dim=input_dim, hyp=hyp, opt=opt, mean=mean, cov=cov, inf=inf, tile_hyp_cov=False,
                            grid_kwargs=grid_kwargs)
        self.gpml.update_data('tr', x, y)
        self.gpml.update_grid('tr')
        if fast:
            return self.gpml.fast_predict(xstar, return_var=True, verbose=1)
        else:
            return self.gpml.predict(xstar, return_var=True, verbose=1)

if __name__ == '__main__':
    pass
