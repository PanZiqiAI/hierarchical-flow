
import torch
import numpy as np
from custom_pkg.basic.visualizers import plt, gradient_colors


########################################################################################################################
# Plot coordinate lines.
########################################################################################################################

def _compute_xy_for_uv(func_flow, device, us, vs=None):
    """
    :param func_flow: 1d -> 2d.
    :param device:
    :param us: Numpy.array.
    :param vs: Numpy.array.
    :return: xs, ys. Numpy.array.
    """
    if vs is None: vs = np.zeros_like(us)
    assert us.shape == vs.shape
    # Get xs, ys.
    x = torch.from_numpy(np.concatenate([us.reshape(-1, 1), vs.reshape(-1, 1)], axis=1)[:, :, None, None]).to(device)
    zs = func_flow(x).cpu().numpy()[:, :, 0, 0]
    xs, ys = zs[:, 0], zs[:, 1]
    # Reshape.
    xs, ys = xs.reshape(*us.shape), ys.reshape(*us.shape)
    # Return
    return xs, ys


def _vis_grids(data2d, linewidth):
    """
    :param data2d: (n_grids, n_grids, 2)
    :param linewidth:
    :return:
    """
    # 1. Along axis 0.
    colors = gradient_colors(data2d.shape[0], change='blue2green')
    for row_index in range(data2d.shape[0]):
        # Plot current row_line
        for col_index in range(1, data2d.shape[1]):
            pairs = data2d[row_index, col_index-1:col_index+1]
            plt.plot(pairs[:, 0], pairs[:, 1], color=colors[row_index], linewidth=linewidth)
    # 2. Along axis 1.
    colors = gradient_colors(data2d.shape[1], change='blue2red')
    for col_index in range(data2d.shape[1]):
        # Plot current col_line
        for row_index in range(1, data2d.shape[0]):
            pairs = data2d[row_index-1:row_index+1, col_index]
            plt.plot(pairs[:, 0], pairs[:, 1], color=colors[col_index], linewidth=linewidth)


def visualize_coordinate_lines(u1, u2, v_radius, n_grids, n_samples_curve, func_flow, device, save_path):
    """ Visualize coordinate lines of flow. """
    if u1 > u2: u1, u2 = u2, u1
    # Init figure.
    plt.figure(dpi=700)
    # ------------------------------------------------------------------------------------------------------------------
    # 1. Visualize coordinate lines.
    # ------------------------------------------------------------------------------------------------------------------
    # (1) Get xs, ys.
    us, vs = np.arange(u1, u2+1e-5, (u2-u1)/n_grids).astype("float32"), np.arange(-v_radius, v_radius+1e-5, 2.0*v_radius/n_grids).astype("float32")[::-1]
    us, vs = np.meshgrid(us, vs)
    xs, ys = _compute_xy_for_uv(func_flow, device, us=us, vs=vs)
    # (2) Visualize.
    _vis_grids(data2d=np.concatenate([xs[:, :, None], ys[:, :, None]], axis=2), linewidth=1.0)
    # ------------------------------------------------------------------------------------------------------------------
    # 2. Visualize manifold (curve).
    # ------------------------------------------------------------------------------------------------------------------
    # (1) Get xs, ys.
    us_curve = np.arange(u1, u2+1e-5, (u2-u1)/n_samples_curve).astype("float32")
    xs_line, ys_line = _compute_xy_for_uv(func_flow, device, us=us_curve)
    # (2) Visualize.
    plt.plot(xs_line, ys_line, color='black', linewidth=5.0)
    # ------------------------------------------------------------------------------------------------------------------
    # Visualize contours
    # ------------------------------------------------------------------------------------------------------------------
    angles = np.arange(0.0, np.pi+1e-5, np.pi/n_samples_curve)
    for r in np.arange(0.0, 2.0+1e-5, 0.1):
        xs, ys = r*np.cos(angles), r*np.sin(angles)
        # Visualize.
        plt.plot(xs, ys, color='gray', linestyle=":", linewidth=1.0)
    """ Saving """
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()
