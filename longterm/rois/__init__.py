import numpy as np

# copying from nely_suite because of an import error
def local_correlations(Y, eight_neighbours=True, swap_dim=False, order_mean=1):
    """
    This function was copied from CaImAn published in Giovannucci et al. eLife, 2019.
    Computes the correlation image for the input dataset Y
    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format
    
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively
    
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front
    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    if order_mean == 0:
        rho = np.ones(np.shape(Y)[1:])
        rho_h = rho_h
        rho_w = rho_w
        rho[:-1, :] = rho[:-1, :] * rho_h
        rho[1:, :] = rho[1:, :] * rho_h
        rho[:, :-1] = rho[:, :-1] * rho_w
        rho[:, 1:] = rho[:, 1:] * rho_w
    else:
        rho[:-1, :] = rho[:-1, :] + rho_h ** (order_mean)
        rho[1:, :] = rho[1:, :] + rho_h ** (order_mean)
        rho[:, :-1] = rho[:, :-1] + rho_w ** (order_mean)
        rho[:, 1:] = rho[:, 1:] + rho_w ** (order_mean)

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d

        neighbors = 6 * np.ones(np.shape(Y)[1:])
        neighbors[0] = neighbors[0] - 1
        neighbors[-1] = neighbors[-1] - 1
        neighbors[:, 0] = neighbors[:, 0] - 1
        neighbors[:, -1] = neighbors[:, -1] - 1
        neighbors[:, :, 0] = neighbors[:, :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)

            if order_mean == 0:
                rho_d1 = rho_d1
                rho_d2 = rho_d2
                rho[:-1, :-1] = rho[:-1, :-1] * rho_d2
                rho[1:, 1:] = rho[1:, 1:] * rho_d1
                rho[1:, :-1] = rho[1:, :-1] * rho_d1
                rho[:-1, 1:] = rho[:-1, 1:] * rho_d2
            else:
                rho[:-1, :-1] = rho[:-1, :-1] + rho_d2 ** (order_mean)
                rho[1:, 1:] = rho[1:, 1:] + rho_d1 ** (order_mean)
                rho[1:, :-1] = rho[1:, :-1] + rho_d1 ** (order_mean)
                rho[:-1, 1:] = rho[:-1, 1:] + rho_d2 ** (order_mean)

            neighbors = 8 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 3
            neighbors[-1, :] = neighbors[-1, :] - 3
            neighbors[:, 0] = neighbors[:, 0] - 3
            neighbors[:, -1] = neighbors[:, -1] - 3
            neighbors[0, 0] = neighbors[0, 0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1, 0] = neighbors[-1, 0] + 1
            neighbors[0, -1] = neighbors[0, -1] + 1
        else:
            neighbors = 4 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 1
            neighbors[-1, :] = neighbors[-1, :] - 1
            neighbors[:, 0] = neighbors[:, 0] - 1
            neighbors[:, -1] = neighbors[:, -1] - 1

    if order_mean == 0:
        rho = np.power(rho, 1.0 / neighbors)
    else:
        rho = np.power(np.divide(rho, neighbors), 1 / order_mean)

    return rho