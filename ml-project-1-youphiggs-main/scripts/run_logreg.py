"""script for generating the best prediction for ML project 1"""

import numpy as np
import matplotlib.pyplot as plt
from utility import *
from proj1_helpers import *


def process_missing(x):
    """process missing values by either drop or replace them with median of the dimension"""
    # replace -999 in replaced_cols
    x[x == -999] = np.nan
    for i in REPLACE_COLS:
        x[:, i][np.isnan(x[:, i])] = np.nanmedian(x[:, i])

    # drop droped_cols
    D = x.shape[1]
    idx = np.setdiff1d(np.arange(D), DROP_COLS)
    x = x[:, idx]

    print("shape after drop: {}".format(x.shape))
    return x


def process_outliers(x, train):
    """process outliers according to the distribution of training set"""
    for i in range(x.shape[1]):
        col = train[:, i]
        std_dev = col.std()
        mean = col.mean()
        x[:, i][np.where((col - mean) > 3 * std_dev)] = mean + 3 * std_dev
        x[:, i][np.where((mean - col) > 3 * std_dev)] = mean - 3 * std_dev

    return x

if __name__ == "__main__":

    # data preprocessing
    print("\n>>> loading data")
    DATA_TRAIN_PATH = "../data/train.csv"
    y, raw_X, ids = load_csv_data(DATA_TRAIN_PATH)
    print("finish loading training data")

    DATA_TEST_PATH = "../data/test.csv"
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print("finish loading test data")

    print("\n>>> process missing data")
    # for missing ratio greather than 70 %, drop the feature
    # others, replace them with median
    DROP_COLS = [4, 5, 6, 12, 26, 27, 28]
    REPLACE_COLS = [0, 23, 24, 25]
    raw_X = process_missing(raw_X)
    tX_test = process_missing(tX_test)

    print("\n>>> process outliers")
    raw_X = process_outliers(raw_X, raw_X)
    tX_test = process_outliers(tX_test, raw_X)

    print("\n>>> standardize")
    # standardize
    tX, mean_x, std_x = standardize(raw_X)
    tX_test = standardize_test(tX_test, mean_x, std_x)

    print("\n>>> feature expansion")
    tX_poly = build_poly(tX, 3)
    tX_poly = add_shift(tX_poly)
    tX_test_poly = build_poly(tX_test, 3)
    tX_test_poly = add_shift(tX_test_poly)

    print("{} data points, {} features in train set".format(tX_poly.shape[0], tX_poly.shape[1]))
    print(
        "{} data points, {} features in test set".format(
            tX_test_poly.shape[0], tX_test_poly.shape[1]
        )
    )

    # process y for logistic regression
    # map y from (-1, 1) to (0,1)
    y[np.where(y == -1)] = 0

    # train the model
    print("\n>>> begin training")
    # max_iter: maximal number of iteration
    # lambda_: coefficient for penalized term
    # gamma: w = w - gamma * grad
    # isplot: whether to plot the optimization process
    # conv_threshold: convergence threshold
    # maxstep: maximal stepsize
    w_opt = logistic_regression_penalized_gradient_descent(
        y, tX_poly, max_iter=501, lambda_=0.1, gamma=0.5, isplot=False, conv_threshold=1e-6, maxstep=0.5
    )

    # evaluate with cross validation
    # k_fold: fold of cross validation
    # threshold: threshold for prediction i.e. pred < threshold -> -1; pred >= threshold -> 1
    # seed: seed for rng in data spliting
    # islinear: if True, then compute mse loss, if False, compute cross entropy loss
    print("\n>>> evaluating with cross validation")
    tr_loss, te_loss, accr = cv_eval_notrain(y, tX_poly, w_opt, 5, 0.5, seed=1, islinear=False)
    print("cross validation result:\n mean train loss {:.2e}\t mean test loss {:.2e}\t mean acccuracy {:.2%}".format(tr_loss, te_loss, accr))

    print("\n>>> generating prediction")
    OUTPUT_PATH = './pred_best_logreg.csv'
    y_pred = pred_logistic(tX_test_poly, w_opt, 0.5, for_submission=True)
    # check validity before submission
    assert len(y_pred) == 568238
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

