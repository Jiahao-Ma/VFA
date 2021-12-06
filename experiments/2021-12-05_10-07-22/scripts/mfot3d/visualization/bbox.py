import numpy as np

def project(corner_3d, calib):
    """ Project corner_3d `nx4 points` in camera rect coord to image2 plane
        Args:
            corner_3d: nx4 `numpy.ndarray`
            calib: camera projection matrix
        Returns:
            corner_2d: nx2 `numpy.ndarray` 2d points in image2
    """
    corner_2d = np.dot(calib, corner_3d.T)
    corner_2d[0, :] = corner_2d[0, :] / corner_2d[2, :]
    corner_2d[1, :] = corner_2d[1, :] / corner_2d[2, :]
    corner_2d = np.array(corner_2d, dtype=np.int)
    return corner_2d[0:2, :].T

def rotz(t):
    """ Rotation about z-axis """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def compute_3d_bbox(dimension, rotation, location, calib):
    h, w, l = dimension[0], dimension[1], dimension[2]
    x = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]
    z = [0, 0, 0, 0, h, h, h, h]
    # NOTICE! rotation: -np.pi ~ np.pi ! instead of -180 ~ 180
    rotMat = rotz(rotation)
    corner_3d = np.vstack([x, y, z])
    corner_3d = np.dot(rotMat, corner_3d)
    bottom_center = np.tile(location, (corner_3d.shape[1], 1)).T
    corner_3d = corner_3d + bottom_center
    corner_3d_homo = np.vstack([corner_3d, np.ones((1, corner_3d.shape[1]))])
    corner_2d = project(corner_3d_homo.T, calib)
    return corner_2d


def draw_3DBBox(ax, dimension, rotation, location, calib, edgecolor=(0, 1, 0), linewidth=1):
    corners = compute_3d_bbox(dimension, rotation, location, calib)
    if len(corners) != 8:
        return ax
    assert corners.shape[1] == 2, 'corners` shape should be [8, 2]'
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        ax.plot((corners[i, 0], corners[j, 0]), (corners[i, 1], corners[j, 1]), color=edgecolor, linewidth=linewidth)
        i, j = k + 4, (k + 1) % 4 + 4
        ax.plot((corners[i, 0], corners[j, 0]), (corners[i, 1], corners[j, 1]), color=edgecolor, linewidth=linewidth)
        i, j = k, k + 4
        ax.plot((corners[i, 0], corners[j, 0]), (corners[i, 1], corners[j, 1]), color=edgecolor, linewidth=linewidth)
    return ax