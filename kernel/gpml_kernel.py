from backend.gpml import GPML
import numpy as np

_gpml_cov_eval = """
[K dK] = feval({cov}, hyp, x, z);
"""


class GPMLKernel(object):
    def __init__(self, cov):
        self.cov = cov
        self.gpml = GPML()

    def evaluate(self, x, z, hyp):
        config = {'cov': "@%s" % self.cov}
        hyp = np.asarray(hyp)
        self.gpml.eng.push('x', x)
        self.gpml.eng.push('z', z)
        self.gpml.eng.push('hyp', hyp)
        self.gpml.eng.eval(_gpml_cov_eval.format(**config), verbose=1)
        return self.gpml.eng.pull('K')

    def gradient(self, x, z, hyp):
        config = {'cov': "@%s" % self.cov}
        hyp = np.asarray(hyp)
        self.gpml.eng.push('x', x)
        self.gpml.eng.push('z', z)
        self.gpml.eng.push('hyp', hyp)
        self.gpml.eng.eval(_gpml_cov_eval.format(**config), verbose=1)
        return self.gpml.eng.pull('dK')


def test():
    cov = 'covSEiso'
    x = np.random.normal(0, 5, (10, 2))
    hyp = [np.log(1)] * 2
    kernel = GPMLKernel(cov)
    K1 = kernel.evaluate(x, x, hyp)
    from seiso import SEiso
    K2 = SEiso.evaluate(x, x, hyp)
    K_err = K1 - K2
    for r in K_err:
        print r


if __name__ == '__main__':
    test()
