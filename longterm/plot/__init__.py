import matplotlib.pyplot as plt
import numpy as np


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
                    roi_signal = stack[:, pixel[0]-roi_size:pixel[0]+roi_size, pixel[1]-roi_size:pixel[1]]
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

