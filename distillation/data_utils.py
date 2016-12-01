import scipy.io as sio
import numpy as np


DATA_PATH = '../data/'
KIN40K = 'kin40k'
PUMADYN32NM = 'pumadyn32nm'
ABALONE = 'abalone'
BOSTON = 'boston'


def load_kin40k():
    data_path = DATA_PATH + KIN40K + '/'
    train_x = np.genfromtxt(data_path + KIN40K + '_train_data.asc', dtype=float)
    train_y = np.genfromtxt(data_path + KIN40K + '_train_labels.asc', dtype=float)
    test_x = np.genfromtxt(data_path + KIN40K + '_test_data.asc', dtype=float)
    test_y = np.genfromtxt(data_path + KIN40K + '_test_labels.asc', dtype=float)
    return train_x, train_y, test_x, test_y


def load_pumadyn32nm():
    data_path = DATA_PATH + PUMADYN32NM + '/'
    train_x = np.genfromtxt(data_path + PUMADYN32NM + '_train_data.asc', dtype=float)
    train_y = np.genfromtxt(data_path + PUMADYN32NM + '_train_labels.asc', dtype=float)
    test_x = np.genfromtxt(data_path + PUMADYN32NM + '_test_data.asc', dtype=float)
    test_y = np.genfromtxt(data_path + PUMADYN32NM + '_test_labels.asc', dtype=float)
    return train_x, train_y, test_x, test_y


def load_abalone():
    fname = DATA_PATH + 'abalone_split.mat'
    train_x = sio.loadmat(fname)['Xtrain']
    train_y = sio.loadmat(fname)['Ytrain'].flatten()
    test_x = sio.loadmat(fname)['Xtest']
    test_y = sio.loadmat(fname)['Ytest'].flatten()
    return train_x, train_y, test_x, test_y


def load_boston():
    fname = DATA_PATH + 'data_boston.mat'
    train_x = sio.loadmat(fname)['X']
    train_y = sio.loadmat(fname)['y'].flatten()
    test_x = sio.loadmat(fname)['Xstar']
    test_y = sio.loadmat(fname)['ystar'].flatten()
    return train_x, train_y, test_x, test_y


def load_dataset(dataset):
    if dataset == PUMADYN32NM:
        return load_pumadyn32nm()
    elif dataset == KIN40K:
        return load_kin40k()
    elif dataset == BOSTON:
        return load_boston()
    elif dataset == ABALONE:
        return load_abalone()
    else:
        raise ValueError(dataset)


if __name__ == '__main__':
    a, _, c, _ = load_boston()
    print a.shape, c.shape
