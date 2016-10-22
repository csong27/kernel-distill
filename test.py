from cov.se_iso import SEiso
import numpy as np
import matplotlib.pyplot as plt


def test():
    x = np.random.normal(0, 5, size=(100, 1))
    x = np.sort(x, axis=0)
    se = SEiso()
    K = se.evaluate(x, x, [np.log(3), np.log(0.5)])
    plt.pcolor(K)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test()
