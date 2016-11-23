import numpy as np


DATA_PATH = '../data/'
KIN40K = 'kin40k'
PUMADYN32NM = 'pumadyn32nm'


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


if __name__ == '__main__':
    pass
