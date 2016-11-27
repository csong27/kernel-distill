import numpy as np


def smae(true_y, pred_y):
    return np.mean(np.abs(pred_y - true_y)) / np.mean(np.abs(true_y - np.mean(true_y)))


def smse(true_y, pred_y):
    return np.mean(np.square(pred_y - true_y)) / np.var(true_y)


def rmse(true_y, pred_y):
    return np.sqrt(np.mean(np.square(pred_y - true_y)))

