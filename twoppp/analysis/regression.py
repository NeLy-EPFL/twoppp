from tqdm import tqdm

import numpy as np
from numpy.random import default_rng
rng = default_rng(seed=5)
from sklearn.isotonic import IsotonicRegression

from twoppp.utils import standardise

def fit_regression(X, y, alpha=0):
    return np.linalg.inv(X.T.dot(X) + alpha * np.eye(X.shape[-1])).dot(X.T).dot(y)

def predict_regression(X, w):
    return X.dot(w)

def fit_predict(X, y, alpha=0):
    w = fit_regression(X,y, alpha=alpha)
    return predict_regression(X,w)

def var_exp(y, y_hat):
    return 1 - np.var(y-y_hat, axis=0)/np.var(y, axis=0)

def fit_predict_varexp(X, y, alpha=0):
    y_hat = fit_predict(X, y, alpha=alpha)
    return var_exp(y, y_hat)

def get_cv_split(N_samples, N_splits=5, random=True):
    samples = np.arange(N_samples)
    N_per_split = N_samples // N_splits
    split = np.zeros((N_samples))
    if random:
        rng.shuffle(samples)
    for i_split in range(N_splits):
        if i_split < N_splits - 1:
            split[samples[i_split*N_per_split:(i_split+1)*N_per_split]] = i_split
        else:
            split[samples[i_split*N_per_split:]] = i_split
    return split

def fit_predict_cv(X,y, N_splits=6, random=False, alpha=1e-3):
    N_samples = X.shape[0]
    splits = get_cv_split(N_samples, N_splits=N_splits, random=random)
    y_hat = np.zeros_like(y)
    ws = []
    for i_split in range(N_splits):
        test_samples = splits == i_split
        train_samples = splits != i_split
        w = fit_regression(X[train_samples], y[train_samples], alpha=alpha)
        y_hat[test_samples] = predict_regression(X[test_samples], w)
        ws.append(w)
    return y_hat, ws

def fit_predict_nested_cv(X,y, N_splits=(6,5), random=False, log10_alpha_range=(-3,7,0.01), verbose=True):
    alphas = 10**np.arange(*log10_alpha_range)
    N_samples = X.shape[0]
    cv_splits = get_cv_split(N_samples, N_splits=N_splits[0], random=random)

    best_alphas = []
    global_ws = []
    global_y_hat = np.zeros_like(y)

    for i_cv_split in tqdm(range(N_splits[0]), disable=not verbose):
        test_samples_cv = cv_splits == i_cv_split
        train_samples_cv = cv_splits != i_cv_split
        alpha_splits = get_cv_split(np.sum(train_samples_cv), N_splits=N_splits[1], random=random)
        y_train_cv = y[train_samples_cv]
        X_train_cv = X[train_samples_cv]

        ves = []
        for i_a, alpha in enumerate(alphas):
            y_hat = np.zeros_like(y_train_cv)
            for i_a_split in range(N_splits[1]):
                test_samples_alpha = alpha_splits == i_a_split
                train_samples_alpha = alpha_splits != i_a_split
                w = fit_regression(X_train_cv[train_samples_alpha], y_train_cv[train_samples_alpha], alpha=alpha)
                y_hat[test_samples_alpha] = predict_regression(X_train_cv[test_samples_alpha], w)
            ves.append(np.mean(var_exp(y_train_cv, y_hat)))

        best_alpha = alphas[np.argmax(ves)]
        best_alphas.append(best_alpha)
        w = fit_regression(X_train_cv, y_train_cv, alpha=best_alpha)
        global_ws.append(w)
        global_y_hat[test_samples_cv] = predict_regression(X[test_samples_cv], w)

    return global_y_hat, global_ws, best_alphas

def fit_predict_varexp_nested_cv(X, y, N_splits=(6,5), random=False, log10_alpha_range=(2.9,3.2,0.01), verbose=False):
    y_hat, _, best_alphas = fit_predict_nested_cv(X, y, N_splits=N_splits, random=random,
                                                  log10_alpha_range=log10_alpha_range, verbose=verbose)
    if verbose:
        print(f"alphas: {best_alphas}")
    return var_exp(y, y_hat)

def fit_predict_varexp_individual_nested_cv(X, y, N_splits=(6,5), random=False, log10_alpha_range=(2.9,3.2,0.01), verbose=False):
    individ_var_exp = np.zeros((X.shape[1], y.shape[1]))

    for i_x in tqdm(range(X.shape[1])):
        this_X = X[:, i_x:i_x+1]
        for i_y in range(y.shape[1]):
            this_y = y[:, i_y:i_y+1]
            this_var_exp = fit_predict_varexp_nested_cv(this_X, this_y,  N_splits=N_splits, random=random,
                                                        log10_alpha_range=log10_alpha_range, verbose=verbose)
            individ_var_exp[i_x, i_y] = this_var_exp
    return individ_var_exp

def shuffle_regressors(X, inds="all"):
    if inds == "all":
        inds = [True]*X.shape[1]
    if (isinstance(inds, list) or isinstance(inds, np.ndarray)) and len(inds) < X.shape[1]:
        inds = [True if i in inds else False for i in range(X.shape[1])]
    return np.hstack([np.random.choice(x, size=x.shape, replace=False)[:, np.newaxis]
                      if i else x[:, np.newaxis] for x, i in zip(X.T, inds)])

def compute_lagged_var_exp(X_binned, y_binned, lags=np.arange(0, 30), separate_vars=True, N_splits=(6, 5),
                           log10_alpha_range=(1.5,3.5,0.5)):
    N_rep, N_t, N_X = X_binned.shape
    # print(N_rep, N_t, N_X)
    N_repy, N_ty, N_y = y_binned.shape
    assert N_rep == N_repy
    assert N_t == N_ty
    N_lags = len(lags)
    
    var_exp_lags = np.zeros((N_lags, N_X, N_y))
    
    for i_lag, lag in enumerate(tqdm(lags)):
        N_shift_possible = N_t - lag
        # print(N_shift_possible)
        X_augment = np.zeros((N_rep*N_shift_possible, N_X))
        y_augment = np.zeros((N_rep*N_shift_possible, N_y))
        # print(X_augment.shape, y_augment.shape)
        for to_shift in range(N_shift_possible):
            # print(to_shift)
            X_augment[to_shift:N_rep*N_shift_possible:N_shift_possible, :] = X_binned[:, to_shift, :]
            y_augment[to_shift:N_rep*N_shift_possible:N_shift_possible, :] = y_binned[:, to_shift+lag, :]
        
        if separate_vars:
            for i_X in range(N_X):
                this_X = standardise(X_augment[:, i_X:i_X+1])
                
                for i_y in range(N_y):
                    this_y = standardise(y_augment[:, i_y:i_y+1])
                    var_exp_lags[i_lag, i_X, i_y] = fit_predict_varexp_nested_cv(X=this_X, y=this_y,
                                                                                 N_splits=N_splits, random=False,
                                                                                 log10_alpha_range=log10_alpha_range,
                                                                                 verbose=False)
        else:
            raise NotImplementedError
            
    return var_exp_lags

def detrend(X, poly=3, iso=True, iso_pre=True, axis=0, t=None):
    if len(X.shape) > 1:
        raise NotImplementedError(f"X can only be 1D or 2D, but it was {len(X.shape)}D")
    elif len(X.shape) == 1:
        X = X[:, np.newaxis]
        squeeze = True
        transpose = False
    elif len(X.shape) == 0:
        squeeze = False
        if axis == 1:
            X = X.T
            transpose = True
        elif axis == 0:
            transpose = False
        else:
            raise NotImplementedError(f"axis must be 0 or 1, but it was {axis}")

    t = np.arange(len(X)) if t is None else t
    X_raw = X.copy()
    if iso and iso_pre:
        X_fit = np.array([IsotonicRegression(increasing="auto").fit_transform(t, X[:, i])
                           for i in range(X.shape[1])]).T
        X -= X_fit
    if poly:
        t_poly = np.hstack([standardise(t[:, np.newaxis]**i) for i in range(poly)])
        X_fit = fit_predict(X=t_poly, y=X)
        X -= X_fit
    if iso and not iso_pre:
        X_fit = np.array([IsotonicRegression(increasing="auto").fit_transform(t, X[:, i])
                           for i in range(X.shape[1])]).T
        X -= X_fit
    fit_var_exp = var_exp(X_raw, X)
    X = np.squeeze(X) if squeeze else X
    X = X.T if transpose else X
    return X, fit_var_exp