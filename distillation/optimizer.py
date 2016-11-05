import numpy as np


class SGD(object):
    def __init__(self, eta):
        self.eta = eta

    def adjust_grad(self, grad):
        return self.eta * grad


class AdaGrad(SGD):
    def __init__(self, eta, shape, fudge_factor=1e-6):
        SGD.__init__(self, eta)
        self.gti = np.zeros(shape)
        self.fudge_factor = fudge_factor

    def adjust_grad(self, grad):
        self.gti += grad ** 2
        adjusted_grad = grad / (self.fudge_factor + np.sqrt(self.gti))
        return self.eta * adjusted_grad


class Adam(SGD):
    def __init__(self, eta, b1=0.1, b2=0.001, e=1e-8):
        SGD.__init__(self, eta)
        self.i = 0.
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.m = None
        self.v = None

    def adjust_grad(self, grad):
        i_t = self.i + 1.
        fix1 = 1. - (1. - self.b1) ** i_t
        fix2 = 1. - (1. - self.b2) ** i_t
        lr_t = self.eta * (np.sqrt(fix2) / fix1)
        if self.m is None:
            self.m = grad * 0.
            self.v = grad * 0.
        m_t = (self.b1 * grad) + ((1. - self.b1) * self.m)
        v_t = (self.b2 * np.square(grad)) + ((1. - self.b2) * self.v)
        grad = lr_t * m_t / (np.sqrt(v_t) + self.e)
        self.i = i_t
        self.eta = lr_t
        self.m = m_t
        self.v = v_t
        return grad
