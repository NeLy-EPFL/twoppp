import os,sys
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as clrs
from scipy.ndimage import gaussian_filter1d
import pickle
import gc

FILE_PATH = os.path.realpath(__file__)
ANALYSIS_PATH, _ = os.path.split(FILE_PATH)
TWOPPP_PATH, _ = os.path.split(ANALYSIS_PATH)
MODULE_PATH, _ = os.path.split(TWOPPP_PATH)
sys.path.append(MODULE_PATH)

from twoppp.plot import confidence_ellipse
from twoppp import utils
from twoppp import rois
from twoppp.behaviour import optic_flow as of
from twoppp.behaviour import synchronisation as sync

def cluster_corr(corr_array):
    """
    Returns indices to rearrange the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to each other 
    Modified from: https://wil.yegelwel.com/cluster-correlation-matrix/ 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a Nx1 index matrix that can be used to re-arrange the rows and columns of the correlation matrix
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    
    return idx

def cov(X, submean=True):
    """
    compute the covariance matrix of X
    """
    X = deepcopy(X)
    if submean:
        X -= X.mean(axis=0)
    return X.T.dot(X) / len(X)

def zscore_X1_X2(X1,X2):
    """
    zscore X1 and X2 based on the mean and std of the combined array X = [X1.T, X2.T].T 
    """
    X1, X2 = deepcopy(X1), deepcopy(X2)
    X = np.vstack((X1, X2))
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    X1 = (X1 - X_mean) / X_std
    X2 = (X2 - X_mean) / X_std
    return X, X1, X2

def cov_pair(X1, X2, zscore=True):
    """
    compute:
    - total covariance of X = [X1.T, X2.T].T: cov_X
    - covariance of X1: cov_X1
    - covariance of X2: cov_X2
    - intra-class covariance: cov_intra
    - inter-class (between classes) covariance: cov_inter
    """
    X1, X2 = deepcopy(X1), deepcopy(X2)
    N1 = len(X1)
    N2 = len(X2)
    N = N1 + N2
    if N == 0:
        return None, None, None, None, None
    if zscore:
        X, X1, X2 = zscore_X1_X2(X1,X2)
    else:
        X = np.vstack((X1, X2))
    cov_X = cov(X)
    cov_X1 = cov(X1)
    cov_X2 = cov(X2)
    cov_intra = N1/N*cov_X1 + N2/N*cov_X2
    cov_inter = cov_X - cov_intra
    # cov_inter = N1/N*np.expand_dims(X1.mean(axis=0), axis=1).dot(np.expand_dims(X1.mean(axis=0), axis=0)) + \
    #             N2/N*np.expand_dims(X2.mean(axis=0), axis=1).dot(np.expand_dims(X2.mean(axis=0), axis=0))
    return cov_X, cov_X1, cov_X2, cov_intra, cov_inter

def all_cov_pairs(X_list):
    """
    compute for all combinations of classes assuming X_list is a list containing a data array for each class/trial:
    - total covariance of X = [X1.T, X2.T].T: cov_total
    - intra-class covariance: cov_intra
    - inter-class (between classes) covariance: cov_inter
    - summary matrix with intra above diagonal and inter below diagonal: cov_matrix
    """
    X_list = deepcopy(X_list)
    N_trials = len(X_list)
    N_neurons = X_list[0].shape[-1]
    cov_intra = np.zeros((N_trials, N_trials, N_neurons, N_neurons))
    cov_inter = np.zeros((N_trials, N_trials, N_neurons, N_neurons))
    cov_total = np.zeros((N_trials, N_trials, N_neurons, N_neurons))
    cov_matrix = np.zeros((N_trials, N_trials, N_neurons, N_neurons))
    for i_1 in range(N_trials):
        cov_matrix[i_1, i_1, :, :] = cov(X_list[i_1])
        if i_1 == N_trials -1:
            break
        for i_2 in range(i_1+1, N_trials):
            _cov_total, _cov_X1, _cov_X2, _cov_intra, _cov_inter = cov_pair(X_list[i_1], X_list[i_2])
            cov_intra[i_1, i_2, :, :] = _cov_intra
            cov_inter[i_1, i_2, :, :] = _cov_inter
            cov_total[i_1, i_2, :, :] = _cov_total
            
            cov_matrix[i_1, i_2, :, :] = _cov_intra
            cov_matrix[i_2, i_1, :, :] = _cov_inter
    return cov_matrix, cov_total, cov_intra, cov_inter

def inter_condition_pca(X1, X2, zscore=False, norm=True, return_norm=False):
    """
    compute weights to project data on axis of highest inter condition variance
    """
    X1, X2 = deepcopy(X1), deepcopy(X2)
    if zscore:
        _, X1, X2 = zscore_X1_X2(X1,X2)
    w = X1.mean(axis=0) - X2.mean(axis=0)
    if norm:
        WW = np.linalg.norm(w)
        w /= WW
        if return_norm:
            return w, WW
    return w

def inter_condition_lda(X1, X2, zscore=False, norm=True, return_norm=False):
    """
    compute weights to project data on axis of best discrimination
    """
    X1, X2 = deepcopy(X1), deepcopy(X2)
    if zscore:
        _, X1, X2 = zscore_X1_X2(X1,X2)
    N1 = len(X1)
    N2 = len(X2)
    N = N1 + N2
    if N == 0:
        if return_norm:
            return None, None
        return None
    cov_intra = N1/N*cov(X1) + N2/N*cov(X2)
    w = np.linalg.inv(cov_intra).dot(X1.mean(axis=0) - X2.mean(axis=0))
    if norm:
        WW = np.linalg.norm(w)
        w /= WW
        if return_norm:
            return w, WW
    return w

def pca(X, n=None, zscore=False):
    if zscore:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    n = X.shape[-1] if n is None else n
    lamd_pca, v_pca = np.linalg.eig(cov(X))
    return v_pca[:,:n]

def all_inter_condition_pairs(X_list, zscore=False, norm=True, return_norm=False, use_pca=True, use_lda=True):
    N_trials = len(X_list)
    N_neurons = X_list[0].shape[-1]
    w_inter_lda = np.zeros((N_trials, N_trials, N_neurons))
    w_inter_pca = np.zeros((N_trials, N_trials, N_neurons))
    w_pca = np.zeros((N_trials, N_trials, N_neurons))
    if return_norm:
        norm_inter_lda = np.zeros((N_trials, N_trials))
        norm_inter_pca = np.zeros((N_trials, N_trials))
    for i_1 in range(N_trials):
        if i_1 == N_trials - 1:
            break
        for i_2 in range(i_1+1,N_trials):
            if return_norm:
                w_inter_pca[i_1, i_2, :], norm_inter_pca[i_1, i_2] = inter_condition_pca(X_list[i_1], X_list[i_2], zscore, norm=True, return_norm=True)
                if use_lda:
                    w_inter_lda[i_1, i_2, :], norm_inter_lda[i_1, i_2] = inter_condition_lda(X_list[i_1], X_list[i_2], zscore, norm=True, return_norm=True)
            else:
                w_inter_pca[i_1, i_2, :] = inter_condition_pca(X_list[i_1], X_list[i_2], zscore, norm=norm)
                if use_lda:
                    w_inter_lda[i_1, i_2, :] = inter_condition_lda(X_list[i_1], X_list[i_2], zscore, norm=norm)
            if use_pca:
                w_pca[i_1, i_2, :] = pca(np.concatenate([X_list[i_1], X_list[i_2]]), n=1, zscore=zscore).squeeze()
    if return_norm:
        return w_inter_pca, w_inter_lda, w_pca, norm_inter_pca, norm_inter_lda
    return w_inter_pca, w_inter_lda, w_pca

def plot_cov_matrix(cov_matrix, trials, name="", figsize=(9.5,10), clim=None):
    N_trials = len(trials)
    fig, axs = plt.subplots(N_trials, N_trials, figsize=figsize, sharex=True, sharey=True)
    for i_row, axs_ in enumerate(axs):
        for i_col, ax in enumerate(axs_):
            im = cov_matrix[trials[i_row], trials[i_col], :, :]
            if clim is None and i_row == 0 and i_col==0:
                clim = [np.quantile(im, 0.01), np.quantile(im, 0.99)]
            ax.imshow(im, cmap=plt.cm.get_cmap("seismic"), clim=clim)
            if i_col == i_row:
                ax.set_title("covariance trial {}".format(trials[i_row]))
            elif i_col > i_row:
                ax.set_title("intra trial {} and {}".format(trials[i_col], trials[i_row]))
            elif i_col < i_row:
                ax.set_title("inter trial {} and {}".format(trials[i_col], trials[i_row]))
    fig.suptitle(name)

def plot_proj_hists(w_matrix, X_trials, trials, name="", X_names=None, figsize=(9.5,10), clim=None, colors=None):
    N_trials = len(trials)
    X_names = trials if X_names is None else X_names
    fig, axs = plt.subplots(N_trials-1, N_trials-1, figsize=figsize)  # , sharex=True, sharey=True)
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:N_trials]
    for i_row, axs_ in enumerate(axs):
        for i_col, ax in enumerate(axs_):
            i_col += 1
            if i_col > i_row:
                w = w_matrix[trials[i_row], trials[i_col]]
                X_proj = [X_trials[trial].dot(w) for trial in trials]
                try:
                    ax.hist(X_proj, bins=20, label=X_names, color=colors)
                except:
                    print("Could not display info for trials {} and {}".format(trials[i_row], trials[i_col]))
                means = [np.mean(X_) for X_ in X_proj]
                # print(means)
                ax.set_prop_cycle(None)
                _ = [ax.scatter(mean, -100, s=100, color=color) for mean, color in zip(means, colors)]
                ax.set_title("dir "+str(X_names[i_row])+" & "+str(X_names[i_col]))
                if i_row == 0 and i_col == 1:
                    ax.legend(frameon=False)  # , bbox_to_anchor=(1, 0.5))
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                # ax.set_yscale("log")
            else:
                ax.axis("off")
            
    fig.suptitle(name)
    fig.tight_layout()

def plot_proj_scatter(w_1, w_2, X_trials, trials, name="", w_names = [None, None], X_names=None, 
                 figsize=(9.5,10), alpha=0.01, fig=None, ax=None, legendpos=None, colors=None, gradual=False):
    N_trials = len(trials)
    X_names = trials if X_names is None else X_names
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # , sharex=True, sharey=True)
    xs = [X_trials[trial].dot(w_1) for trial in trials]
    ys = [X_trials[trial].dot(w_2) for trial in trials]
    X_names = [None for trial in trials] if X_names is None else X_names
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:N_trials]
    for x,y,x_name, color in zip(xs, ys, X_names, colors):
        try:
            if gradual:
                N_samples = len(x)
                color_scale = [np.minimum(np.array(clrs.to_rgb(color)) * f, 1) for f in np.linspace(0.75, 1.2, N_samples)]
                ax.scatter(x, y, alpha=alpha, color=color_scale)
            else:
                ax.scatter(x,y, alpha=alpha, color=color)
        except:
            print("could not display data ", X_names)
    for x,y,x_name, color in zip(xs, ys, X_names, colors):
        try:
            ax.scatter(np.mean(x),np.mean(y), label=x_name, alpha=1, s=100, color=color, edgecolors="white")
            confidence_ellipse(x,y, ax=ax, color=color)
        except:
            print("could not display data ", X_names)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legendpos is None:
        ax.legend(frameon=False)
    else:
        ax.legend(frameon=False, loc=legendpos)
    ax.set_xlabel(w_names[0])
    ax.set_ylabel(w_names[1])
            
    ax.set_title(name)
    try:
        fig.tight_layout()
        pass
    except:
        print("could not apply tight layout.")

def inter_condition_variance_ratios(X_walk, X_rest, zscore=True):
    pass
    # input: X_1 = [X_walk1, X_walk2], X_2 = [X_rest1, X_rest2]
    N_walk = [len(X_) for X_ in X_walk]
    N_rest = [len(X_) for X_ in X_rest]
    N_neurons = X_walk[0].shape[1]
    # 1. zscore such that np.var(X) = 1
    if zscore:
        X, X_walk, X_rest = zscore_X1_X2(np.concatenate(X_walk), np.concatenate(X_rest))
    else:
        X_walk = np.concatenate(X_walk)
        X_rest = np.concatenate(X_rest)
        X = np.concatebate([X_walk, X_rest])
    # 2. compute X_inter_walk_rest
    _, _, _, _, cov_inter = cov_pair(X_walk, X_rest, zscore=False)
    # 3. variance explained walk rest = ||X_inter_walk_rest|| / N_neurons
    var_exp_walk_rest = np.linalg.norm(cov_inter) / N_neurons
    # 4. take X_walk, do NOT z-score, compute X_inter_walk
    _, _, _, _, cov_inter_walk = cov_pair(X_walk[:N_walk[0], :], X_walk[N_walk[0]:, :], zscore=False)
    # 5. variance explained walk time = ||X_inter_walk|| / N_neurons
    var_exp_walk_time = np.linalg.norm(cov_inter_walk) / N_neurons * np.sum(N_walk) / len(X)
    # 6. take X_rest, do NOT z-score, compute X_inter_rest
    _, _, _, _, cov_inter_rest = cov_pair(X_rest[:N_rest[0], :], X_rest[N_rest[0]:, :], zscore=False)
    # 7. variance explained rest time = ||X_inter_rest|| / N_neurons
    var_exp_rest_time = np.linalg.norm(cov_inter_rest) / N_neurons * np.sum(N_rest) / len(X)
    # 8. compute weighted ratio
    ratio = (var_exp_walk_time + var_exp_rest_time) / var_exp_walk_rest
    return ratio, var_exp_walk_rest, var_exp_walk_time, var_exp_rest_time

def all_inter_condition_variance_ratios(X_walks, X_rests, zscore=True):
    N_trials = len(X_walks)
    N_neurons = X_walks[0].shape[-1]
    ratios = np.zeros((N_trials, N_trials))
    var_exp_walk_rests = np.zeros((N_trials, N_trials))
    var_exp_walk_times = np.zeros((N_trials, N_trials))
    var_exp_rest_times = np.zeros((N_trials, N_trials))
    for i_1 in range(N_trials):
        ratios[i_1, i_1], var_exp_walk_rests[i_1, i_1], var_exp_walk_times[i_1, i_1], var_exp_rest_times[i_1, i_1] = \
            inter_condition_variance_ratios(X_walk=np.array_split(X_walks[i_1], 2, axis=0),
                                            X_rest=np.array_split(X_rests[i_1], 2, axis=0),
                                            zscore=zscore)
        if i_1 == N_trials - 1:
            break
        for i_2 in range(i_1+1,N_trials):
            ratios[i_1, i_2], var_exp_walk_rests[i_1, i_2], var_exp_walk_times[i_1, i_2], var_exp_rest_times[i_1, i_2] = \
                inter_condition_variance_ratios(X_walk=[X_walks[i_1], X_walks[i_2]],
                                                X_rest=[X_rests[i_1], X_rests[i_2]],
                                                zscore=zscore)
    return ratios, var_exp_walk_rests, var_exp_walk_times, var_exp_rest_times
    
class InterPCAAnalysis():
    def __init__(self, fly_dir, i_trials, condition, compare_i_trials=None, thres_walk=0.03, thres_rest=0.01,
                 load_df=True, load_pixels=False, pixel_shape=None, sigma=3, trial_names=None, 
                 twop_df_name="twop_df.pkl", green_stack_name="green_denoised_t1.tif",
                 opflow_df_name="opflow_df.pkl", roi_center_file="ROI_centers.txt", zscore_trials="all"):
        self.fly_dir = fly_dir
        self.all_trial_dirs = utils.readlines_tolist(os.path.join(self.fly_dir, "trial_dirs.txt"))
        self.i_trials = i_trials
        self.trial_names = trial_names
        self.condition = condition
        self.compare_i_trials = compare_i_trials if compare_i_trials is not None else i_trials
        self.thres_walk = thres_walk
        self.thres_rest = thres_rest
        self.sigma = sigma
        self.pixel_shape = pixel_shape
        self.selected_pixels = None
        self.neurons_i_sort = None
        self.pixels_i_sort = None
        self.twop_df_name = twop_df_name
        self.green_stack_name = green_stack_name
        self.opflow_df_name = opflow_df_name
        self.roi_center_file = roi_center_file
        self.zscore_trials = zscore_trials

        self.load(load_df=load_df, load_pixels=load_pixels)

    def load(self, load_df=True, load_pixels=False, pixel_shape=None):
        try:
            self.roi_center = rois.read_roi_center_file(os.path.join(self.fly_dir, "processed", self.roi_center_file))
        except:
            print("Could not load ROIs for fly ", self.fly_dir)
            self.roi_center = None

        self.opflow_dfs = [pd.read_pickle(os.path.join(processed_dir, self.opflow_df_name))
                           for processed_dir in self.processed_dirs]
        # filtering already performed when saving dataframes
        # self.opflow_dfs = [of.filter_velocities(opflow_df) for opflow_df in self.opflow_dfs]
        self.preprocess_rest_walk()

        if load_df:
            self.neural_dfs = [pd.read_pickle(os.path.join(processed_dir, self.twop_df_name))
                               for processed_dir in self.processed_dirs]
            self.preprocess_neurons()
            self.split_walk_rest_neurons()
        else:
            self.neural_dfs = None
            self.neurons_mean = None
            self.neurons_std = None
            self.neurons = None
            self.neurons_i_sort = None
            self.neurons_walk = None
            self.neurons_rest = None

        if load_pixels:
            self.pixel_shape = self.pixel_shape if pixel_shape is None else pixel_shape
            neural_datas = [utils.get_stack(os.path.join(processed_dir, self.green_stack_name)) 
                     for processed_dir in self.processed_dirs]
            if self.pixel_shape is not None:
                neural_datas = [neural_data[:, self.pixel_shape[0]:self.pixel_shape[1], 
                                            self.pixel_shape[2]: self.pixel_shape[3]] for neural_data in neural_datas]
            else:
                self.pixel_shape = [0, neural_datas[0].shape[1], 0, neural_datas[0].shape[2]]
            self.pixels_raw = [np.reshape(neural_data, (neural_data.shape[0], -1)) 
                      for neural_data in neural_datas]
            del neural_datas
            self.preprocess_pixels()
            self.split_walk_rest_pixels()
            self.pixels_i_sort = np.arange(self.pixels[0].shape[1])
        else:
            self.pixel_shape = None
            self.pixels_raw = None
            self.pixels_mean = None
            self.pixels_std = None
            self.pixels = None
            self.pixels_i_sort = None
            self.pixels_walk = None
            self.pixels_rest = None

    def preprocess_rest_walk(self, thres_walk=None, thres_rest=None):
        if thres_walk is None:
            thres_walk = self.thres_walk
        if thres_rest is None:
            thres_rest = self.thres_rest    
        self.v_f = [sync.reduce_during_2p_frame(twop_index=opflow_df.twop_index, 
                                             values=np.expand_dims(opflow_df.velForw, axis=1))[:-1].squeeze() 
                 for opflow_df in self.opflow_dfs]
        self.v_s = [sync.reduce_during_2p_frame(twop_index=opflow_df.twop_index, 
                                                    values=np.expand_dims(opflow_df.velSide, axis=1))[:-1].squeeze() 
                        for opflow_df in self.opflow_dfs]
        self.v_t = [sync.reduce_during_2p_frame(twop_index=opflow_df.twop_index, 
                                                    values=np.expand_dims(opflow_df.velTurn, axis=1))[:-1].squeeze() 
                        for opflow_df in self.opflow_dfs]
        
        rest_reds = [np.logical_and.reduce((np.abs(v_f_red) <= thres_rest, np.abs(v_s_red) <= thres_rest, np.abs(v_t_red) <= thres_rest)) 
                  for v_f_red, v_s_red, v_t_red in zip(self.v_f, self.v_s, self.v_t)]
        rest_strict_reds = [np.logical_and(np.convolve(rest_red, np.ones(16)/16, mode="same") >= 0.75, rest_red) 
                                for rest_red in rest_reds] 
        self.rest = rest_strict_reds
        walk_reds = [v_f_red >= thres_walk for v_f_red in self.v_f] 
        walk_strict_reds = [np.logical_and(np.convolve(walk_red, np.ones(4)/4, mode="same") >= 0.75, walk_red) 
                                for walk_red in walk_reds]
        self.walk = walk_strict_reds

    def preprocess_neurons(self, zscore_trials=None):
        all_data = [neural_df.filter(regex="neuron").to_numpy() for neural_df in self.neural_dfs]
        if self.sigma > 0:
            all_data = [gaussian_filter1d(data, sigma=self.sigma, axis=0) for data in all_data]
        if zscore_trials == "all":
            zscore_trials = np.arange(len(self.i_trials))
        elif zscore_trials == "compare":
            zscore_trials = self.i_compare_in_i_trials
        elif zscore_trials is None:
            zscore_trials = self.zscore_trials
        elif not isinstance(zscore_trials, list):
            raise NotImplementedError
        selected_data = np.concatenate([all_data[i] for i in zscore_trials])
        self.neurons_mean = np.mean(selected_data, axis=0)
        self.neurons_std = np.std(selected_data, axis=0)
        self.neurons = [(data - self.neurons_mean) / self.neurons_std for data in all_data]

    def preprocess_pixels(self, zscore_trials=None):
        all_data = self.pixels_raw
        del self.pixels_raw
        if self.sigma > 0:
            all_data = [gaussian_filter1d(data, sigma=self.sigma, axis=0) for data in all_data]
        if zscore_trials == "all":
            zscore_trials = np.arange(len(self.i_trials))
        elif zscore_trials == "compare":
            zscore_trials = self.i_compare_in_i_trials
        elif zscore_trials is None:
            zscore_trials = self.zscore_trials
        elif not isinstance(zscore_trials, list):
            raise NotImplementedError
        # selected_data = np.concatenate([all_data[i] for i in zscore_trials])
        means = []
        squaremeans = []
        for i in zscore_trials:
            mean = np.mean(all_data[i], axis=0)
            means.append(mean)
        self.pixels_mean = np.mean(means, axis=0)
        for i in zscore_trials:
            squaremean = np.mean(np.square(all_data[i]-self.pixels_mean), axis=0)
            squaremeans.append(squaremean)
        self.pixels_std = np.sqrt(np.mean(squaremeans, axis=0))
        # self.pixels_mean = np.mean(selected_data, axis=0)
        # self.pixels_std = np.std(selected_data, axis=0)
        self.pixels = []
        for i_d, data in enumerate(all_data):
            self.pixels.append((data - self.pixels_mean) / self.pixels_std)
            all_data[i_d] = None
            gc.collect()
        del all_data
        # self.pixels = [(data - self.pixels_mean) / self.pixels_std for data in all_data]
        

    def split_walk_rest_neurons(self):
        self.neurons_walk = [neuron[walk, :] for neuron, walk in zip(self.neurons, self.walk)] 
        self.neurons_rest = [neuron[rest, :] for neuron, rest in zip(self.neurons, self.rest)]

    def split_walk_rest_pixels(self):
        self.pixels_walk = [neuron[walk, :] for neuron, walk in zip(self.pixels, self.walk)]
        self.pixels_rest = [neuron[rest, :] for neuron, rest in zip(self.pixels, self.rest)] 

    def select_pixels(self, selected_pixels):
        self.selected_pixels = selected_pixels
        self.pixels = [pixel[:, self.selected_pixels] for pixel in self.pixels]
        self.split_walk_rest_pixels()

    def sort_neurons(self, i_sort=None):
        if i_sort is not None:
            self.neurons_i_sort = i_sort
        else:
            cov_total = cov(np.concatenate(self.neurons))
            self.neurons_i_sort = cluster_corr(cov_total)
        self.neurons = [neuron[:, self.neurons_i_sort] for neuron in self.neurons]
        self.split_walk_rest_neurons()

    def sort_pixels(self, i_sort=None):
        if i_sort is not None:
            self.pixels_i_sort = i_sort
        else:
            cov_total = cov(np.concatenate(self.pixels))
            self.pixels_i_sort = cluster_corr(cov_total)
        self.pixels = [pixel[:, self.pixels_i_sort] for pixel in self.pixels]
        self.split_walk_rest_pixels()

    def get_w_inter_pca_walk_rest_neurons(self, i_trials="all"):
        if i_trials == "all":
            i_trials = np.arange(len(self.i_trials))
        elif i_trials == "compare":
            i_trials = self.i_compare_in_i_trials
        elif not isinstance(i_trials, list):
            raise NotImplementedError
        data_walk = np.concatenate([self.neurons_walk[i] for i in i_trials])
        data_rest = np.concatenate([self.neurons_rest[i] for i in i_trials])
        w, norm = inter_condition_pca(data_walk, data_rest, norm=True, return_norm=True)
        self.w_inter_pca_walk_rest_neurons = w
        self.norm_inter_pca_walk_rest_neurons = norm
        return w

    def get_w_inter_pca_walk_rest_pixels(self, i_trials="all"):
        if i_trials == "all":
            i_trials = np.arange(len(self.i_trials))
        elif i_trials == "compare":
            i_trials = self.i_compare_in_i_trials
        elif not isinstance(i_trials, list):
            raise NotImplementedError
        data_walk = np.concatenate([self.pixels_walk[i] for i in i_trials])
        data_rest = np.concatenate([self.pixels_rest[i] for i in i_trials])
        w, norm = inter_condition_pca(data_walk, data_rest, norm=True, return_norm=True)
        self.w_inter_pca_walk_rest_pixels = w
        self.norm_inter_pca_walk_rest_pixels = norm
        return w

    def get_covs_walk_rest_neurons(self, i_trials="all"):
        if i_trials == "all":
            i_trials = np.arange(len(self.i_trials))
        elif i_trials == "compare":
            i_trials = self.i_compare_in_i_trials
        elif not isinstance(i_trials, list):
            raise NotImplementedError
        data_walk = np.concatenate([self.neurons_walk[i] for i in i_trials])
        data_rest = np.concatenate([self.neurons_rest[i] for i in i_trials])
        cov_X, cov_X_walk, cov_X_rest, cov_X_intra, cov_X_inter = cov_pair(data_walk, data_rest)
        self.cov_neurons = cov_X
        self.cov_neurons_walk = cov_X_walk
        self.cov_neurons_rest = cov_X_rest
        self.cov_neurons_rest_walk_inter = cov_X_inter
        self.cov_neurons_rest_walk_intra = cov_X_intra

    def get_covs_walk_rest_pixels(self, i_trials="all"):
        if i_trials == "all":
            i_trials = np.arange(len(self.i_trials))
        elif i_trials == "compare":
            i_trials = self.i_compare_in_i_trials
        elif not isinstance(i_trials, list):
            raise NotImplementedError
        data_walk = np.concatenate([self.pixels_walk[i] for i in i_trials])
        data_rest = np.concatenate([self.pixels_rest[i] for i in i_trials])
        cov_X, cov_X_walk, cov_X_rest, cov_X_intra, cov_X_inter = cov_pair(data_walk, data_rest)
        self.cov_pixels = cov_X
        self.cov_pixels_walk = cov_X_walk
        self.cov_pixels_rest = cov_X_rest
        self.cov_pixels_rest_walk_inter = cov_X_inter
        self.cov_pixels_rest_walk_intra = cov_X_intra

    def get_covs_inter_trial_neurons(self):
        cov_matrix_walk, cov_total_walk, cov_intra_walk, cov_inter_walk = all_cov_pairs(self.neurons_walk)
        cov_matrix_rest, cov_total_rest, cov_intra_rest, cov_inter_rest = all_cov_pairs(self.neurons_rest)
        self.cov_neurons_matrix_walk = cov_matrix_walk
        self.cov_neurons_matrix_rest = cov_matrix_rest

    def get_covs_inter_trial_pixels(self):
        cov_matrix_walk, cov_total_walk, cov_intra_walk, cov_inter_walk = all_cov_pairs(self.pixels_walk)
        cov_matrix_rest, cov_total_rest, cov_intra_rest, cov_inter_rest = all_cov_pairs(self.pixels_rest)
        self.cov_pixels_matrix_walk = cov_matrix_walk
        self.cov_pixels_matrix_rest = cov_matrix_rest

    def get_inter_pca_trials_neurons(self, norm=True):
        w_inter_pca_walk, w_inter_lda_walk, w_pca_walk, norm_pca_walk, norm_lda_walk = all_inter_condition_pairs(self.neurons_walk, 
                                                                                        norm=norm, return_norm=True, use_pca=False)
        self.w_inter_pca_walk_neurons = w_inter_pca_walk
        self.w_inter_lda_walk_neurons = w_inter_lda_walk
        self.norm_pca_walk_neurons = norm_pca_walk
        self.norm_lda_walk_neurons = norm_lda_walk
        w_inter_pca_rest, w_inter_lda_rest, w_pca_rest, norm_pca_rest, norm_lda_rest = all_inter_condition_pairs(self.neurons_rest, 
                                                                                  norm=norm, return_norm=True, use_pca=False)
        self.w_inter_pca_rest_neurons = w_inter_pca_rest
        self.w_inter_lda_rest_neurons = w_inter_lda_rest
        self.norm_pca_rest_neurons = norm_pca_rest
        self.norm_lda_rest_neurons = norm_lda_rest

    def get_inter_pca_trials_pixels(self, norm=True):
        w_inter_pca_walk, w_inter_lda_walk, w_pca_walk, norm_pca_walk, norm_lda_walk = all_inter_condition_pairs(self.pixels_walk, 
                                                                                        norm=norm, return_norm=True, 
                                                                                        use_pca=False, use_lda=False)
        self.w_inter_pca_walk_pixels = w_inter_pca_walk
        # self.w_inter_lda_walk_pixels = w_inter_lda_walk
        self.norm_pca_walk_pixels = norm_pca_walk
        # self.norm_lda_walk_pixels = norm_lda_walk
        w_inter_pca_rest, w_inter_lda_rest, w_pca_rest, norm_pca_rest, norm_lda_rest = all_inter_condition_pairs(self.pixels_rest, 
                                                                                  norm=norm, return_norm=True, 
                                                                                  use_pca=False, use_lda=False)
        self.w_inter_pca_rest_pixels = w_inter_pca_rest
        # self.w_inter_lda_rest_pixels = w_inter_lda_rest
        self.norm_pca_rest_pixels = norm_pca_rest
        # self.norm_lda_rest_pixels = norm_lda_rest

    def get_inter_condition_variance_ratios_neurons(self, zscore=True):
        self.ratios_neurons, self.var_exp_walk_rests_neurons, self.var_exp_walk_times_neurons, self.var_exp_rest_times_neurons = \
            all_inter_condition_variance_ratios(X_walks = self.neurons_walk,
                                                X_rests = self.neurons_rest,
                                                zscore=zscore)
    
    def get_inter_condition_variance_ratios_pixels(self, zscore=True):
        self.ratios_pixels, self.var_exp_walk_rests_pixels, self.var_exp_walk_times_pixels, self.var_exp_rest_times_pixels = \
            all_inter_condition_variance_ratios(X_walks = self.pixels_walk,
                                                X_rests = self.pixels_rest,
                                                zscore=zscore)

    def pixels_to_image(self, pixels):
        if self.selected_pixels is not None:
            tmp = np.zeros_like(self.selected_pixels).astype(np.float32)
            tmp[self.selected_pixels] = pixels[np.argsort(self.pixels_i_sort)]
            tmp[np.logical_not(self.selected_pixels)] = None
        else:
            tmp = pixels
        return tmp.reshape(self.size_y,-1)

    def pickle_self(self, out_dir):
        out = {
            "fly_dir": self.fly_dir,
            "i_trials": self.i_trials,
            "compare_i_trials": self.compare_i_trials, 
            "condition": self.condition,
            "trial_names": self.trial_names,
            "sigma": self.sigma,
            "neurons": self.neurons,
            "neurons_mean": self.neurons_mean,
            "neurons_std": self.neurons_std,
            "neurons_i_sort": self.neurons_i_sort,
            "pixels": self.pixels,
            "pixels_mean": self.pixels_mean,
            "pixels_std": self.pixels_std,
            "selected_pixels": self.selected_pixels,
            "pixels_i_sort": self.pixels_i_sort,
            "pixel_shape": self.pixel_shape,
            "walk": self.walk,
            "rest": self.rest,
            "roi_center": self.roi_center,
        }
        with open(out_dir, "wb") as f:
            pickle.dump(out, f)

    def plot_maps(self, map_type="rest", quant=0.99, same_lim=True):
        if map_type == "rest":
            W = self.w_inter_pca_rest_pixels
        elif map_type == "walk":
            W = self.w_inter_pca_walk_pixels
        elif map_type == "walkrest":
            W = np.zeros((2,2,len(self.w_inter_pca_walk_rest_pixels)))
            W[0,1,:] = self.w_inter_pca_walk_rest_pixels
        else:
            return

        N_trials = W.shape[0]
        clims = []

        fig, axs = plt.subplots(N_trials-1, N_trials-1, figsize=(5*(N_trials-1), 3.5*(N_trials-1)), squeeze=False)
        for i_col, axs_ in enumerate(axs):
            for i_row, ax in enumerate(axs_):
                if i_row < i_col:
                    ax.axis("off")
                    continue
                else:
                    i_1 = i_col
                    i_2 = i_row + 1
                W_toshow = self.pixels_to_image(W[i_1,i_2])
                clim = np.quantile(W_toshow, quant)
                clims.append(clim)
                ax.imshow(W_toshow, cmap=plt.cm.get_cmap("seismic"), clim=[-clim, clim])
                if not same_lim and not map_type == "walkrest":
                    ax.set_title(f"{self.trial_names[i_1]} vs {self.trial_names[i_2]}\n\nclim: {clim}")
                elif not map_type == "walkrest":
                    ax.set_title(f"{self.trial_names[i_1]} vs {self.trial_names[i_2]}")
                else:
                    ax.set_title(f"clim: {clim}")
        if same_lim:
            clim = np.mean(clims)
            for ax in axs.flatten():
                if len(ax.images):
                    ax.images[0].set_clim(-clim, clim)
            if not map_type == "walkrest":
                axs[np.min([1,N_trials-1]),0].set_title(f"clim: {clim}")
        fig.suptitle(f"{map_type}" + (" same limit" if same_lim else ""))
        return fig

    def save_maps(self, out_path):
        self.get_w_inter_pca_walk_rest_pixels()
        self.get_inter_pca_trials_pixels()
        with PdfPages(out_path) as pdf:
            for map_type in ["rest", "walk", "walkrest"]:
                for same_lim in [True, False]:
                    # try:
                    fig = self.plot_maps(map_type=map_type, same_lim=same_lim)
                    # except:
                    #     fig = plt.figure(figsize=(1,1))
                    #     print(f"could not generate {map_type} map.")
                    pdf.savefig(fig)
                    plt.close(fig)





    @property
    def i_compare_in_i_trials(self):
        return [np.argwhere(np.array(self.i_trials) == selected)[0,0] 
                for selected in self.compare_i_trials]

    @property
    def trial_dirs(self):
        return [self.all_trial_dirs[i_trial] for i_trial in self.i_trials]

    @property
    def processed_dirs(self):
        return [os.path.join(trial_dir, "processed") for trial_dir in self.trial_dirs]

    @property
    def compare_trial_dirs(self):
        return [self.all_trial_dirs[i_trial] for i_trial in self.compare_i_trials]

    @property
    def compare_processed_dirs(self):
        return [os.path.join(trial_dir, "processed") for trial_dir in self.compare_trial_dirs]

    @property
    def size_y(self):
        return self.pixel_shape[1] - self.pixel_shape[0]

    @property
    def size_x(self):
        return self.pixel_shape[3] - self.pixel_shape[2]

class InterPCAAnalysisFromFile(InterPCAAnalysis):
    def __init__(self, pickle_dir):
        # super.__init__(self)
        print("loading from file: ", pickle_dir)
        self.pickle_dir = pickle_dir
        self.selected_pixels = None
        self.load_from_pickle()
        self.all_trial_dirs = utils.readlines_tolist(os.path.join(self.fly_dir, "trial_dirs.txt"))
        if self.neurons is not None:
            self.split_walk_rest_neurons()
        if self.pixels is not None:
            self.split_walk_rest_pixels()
    
    def load_from_pickle(self):
        with open(self.pickle_dir, "rb") as f:
            data = pickle.load(f)

        self.fly_dir = data["fly_dir"]
        self.i_trials = data["i_trials"]
        self.compare_i_trials = data["compare_i_trials"]
        self.condition = data["condition"]
        self.trial_names = data["trial_names"]
        self.sigma = data["sigma"]
        self.neurons = data["neurons"]
        self.neurons_mean = data["neurons_mean"]
        self.neurons_std = data["neurons_std"]
        self.neurons_i_sort = data["neurons_i_sort"]
        self.pixels = data["pixels"]
        self.pixels_mean = data["pixels_mean"]
        self.pixels_std = data["pixels_std"]
        self.pixels_i_sort = data["pixels_i_sort"]
        self.pixel_shape = data["pixel_shape"]
        self.walk = data["walk"]
        self.rest = data["rest"]
        self.roi_center = data["roi_center"]