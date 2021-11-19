"""
sub-module adding project-spcific plotting functionality and video functions.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np

DARKRED = "#9c0d0b"
DARKORANGE = "#d17c04"
DARKYELLOW = "#dbd80b"
DARKGREEN = "#0e6117"
DARKGREEN_CONTRAST = "#2be031"
DARKCYAN = "#0bbfb3"
DARKBLUE = "#0f0b87"
DARKBLUE_CONTRAST = "#18d5db"
DARKPURPLE = "#730d71"
DARKPINK = "#c20c6d"
DARKBROWN = "#6e4220"
DARKGRAY = "#7d7b79"
BLACK = "#000000"
DARKPLOT = [DARKBLUE, DARKORANGE, DARKGREEN, DARKRED, DARKCYAN, DARKYELLOW, DARKPURPLE, DARKPINK]


def plot_sample_pixels(stacks, pixels, legends=[], colors=[], roi_size=2, f_s=16, figsize=None, alpha=1, spine_outward_shift=3):
    if not isinstance(stacks, list):
        stacks = [stacks]
    N_t, N_y, N_x = stacks[0].shape
    N_stacks = len(stacks)
    if len(legends) == 0:
        legens = [None for _ in range(N_stacks)]
    if len(colors) == 0:
        colors = ["k"]
        _ = [colors.append(plt.cm.Reds(0.1+0.9*(i_s+1)/(N_stacks-1))) for i_s in range(N_stacks - 1)]
    N_pixels = len(pixels)
    x = np.arange(N_t) / f_s

    figsize = (2.5*N_pixels, 10) if figsize is None else figsize
    fig, axs = plt.subplots(ncols=1, nrows=N_pixels, figsize=figsize, sharex=True, sharey=False)

    for i_ax, (ax, pixel) in enumerate(zip(axs, pixels)):
        for i_s, (stack, legend, color) in enumerate(zip(stacks, legends, colors)):
            if roi_size == 0:
                roi_signal = stack[:, pixel[0], pixel[1]]
            else:
                roi_signal = stack[:, pixel[0]-roi_size:pixel[0]+roi_size, pixel[1]-roi_size:pixel[1]+roi_size]
            ax.plot(x, np.mean(roi_signal, axis=(1,2)), color=color, alpha=alpha, label=legend)
        
        ax.set_title(pixel)
        if i_ax == 0:
            ax.legend()
            
        ax.spines['left'].set_position(('outward', spine_outward_shift))  # ('axes', -0.02))  # 'zero'

        # turn off the right spine/ticks
        ax.spines['right'].set_color('none')
        ax.yaxis.tick_left()

        # set the y-spine
        ax.spines['bottom'].set_position(('outward', spine_outward_shift))  # ('axes', -0.02))  # 'zero'

        # turn off the top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()
            
    ax.set_xlabel("time (s)")

    return fig

def confidence_ellipse(x, y, ax, color, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    # modified from https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=color,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def shade_walk_rest(walk, rest, x=None, ax=None, alpha=0.2, colors=["red", "blue"]):
    """shade ranges of the x axis in two different colours

    Parameters
    ----------
    walk : numpy array
        binary array indicating whether fly is walking or not

    rest : numpy array
        binary array indicating whether fly is resting or not

    x : numpy array, optional
        x values that other data on the axes will be/was plotted against.
        if not specified: x = np.arange(len(walk)), by default None

    ax : matplotlib.pyplot.Axes, optional
        axis to be plotted on. if not specified plt.gca(), by default None

    alpha : float, optional
        transparency of shaded area, by default 0.2

    colors : list, optional
        colors to shade walking and resting in, by default ["red", "blue"]
    """
    ax = plt.gca() if ax is None else ax
    x = np.arange(len(walk)) if x is None else x
    
    walk_diff = np.diff(walk.astype(np.int))
    walk_diff_start = np.where(walk_diff==1)[0]
    walk_diff_end = np.where(walk_diff==-1)[0]

    rest_diff = np.diff(rest.astype(np.int))
    rest_diff_start = np.where(rest_diff==1)[0]
    rest_diff_end = np.where(rest_diff==-1)[0]
    N_walk = np.sum(walk_diff==1)
    N_rest = np.sum(rest_diff==1)

    for i_stim in range(N_walk):
        ax.axvspan(x[walk_diff_start[i_stim]], x[walk_diff_end[i_stim]], 
                   alpha=alpha, color=colors[0], ec=None, label="walk" if i_stim==0 else None)
    for i_stim in range(N_rest):
        ax.axvspan(x[rest_diff_start[i_stim]], x[rest_diff_end[i_stim]], 
                   alpha=alpha, color=colors[1], ec=None, label="rest" if i_stim==0 else None)

def plot_mu_sem(mu, err, x=None, label="", alpha=0.3, color=None, ax=None, linewidth=1):
    """
    plot mean and standard deviation,

    Parameters
    ----------
    mu: numpy array
        mean, shape [N_samples, N_lines] or [N_samples]

    err: numpy array
        error to be plotted, e.g. standard error of the mean,
        shape [N_samples, N_lines] or [N_samples]

    x: numpy array, optional
        shape [N_samples]. If not specified will be np.arange(mu.shape[0]),
        by default None

    label: str, optional
        the label for each line either a string if only one line or
        list of strings if multiple lines, by default ""

    alpha: float, optional
        transparency of the shaded area, by default 0.3

    color: optional
        pre-specify colour. if None, use Python default colour cycle, by default None

    ax: matplotlib.pyplot.Axes, optional
        axis to be plotted on, otherwise it will get the current axis with plt.gca(), by default None
    """
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = np.arange(mu.shape[0])
    p = ax.plot(x, mu, lw=linewidth, label=label, color=color)
    if len(mu.shape) is 1:
        ax.fill_between(x, mu - err, mu + err, alpha=alpha, facecolor=p[0].get_color(), edgecolor=None)
    else:
        for i in np.arange(mu.shape[1]):
            ax.fill_between(x, mu[:, i] - err[:, i], mu[:, i] + err[:, i],
                            alpha=alpha, color=p[i].get_color())
