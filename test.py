from cov.se_iso import SEiso
import numpy as np
import matplotlib.pyplot as plt


def test():
    x = np.random.normal(0, 5, size=(500, 1))
    x = np.sort(x, axis=0)
    se = SEiso()
    K = se.evaluate(x, x, [np.log(5), np.log(1.)])
    plt.pcolor(K)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    test()
