import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from moft.utils import to_numpy, grid_rot180
from .bbox import draw_3DBBox

def visualize_image(image):
    # image format: (3, H, W) value range: (0, 1)
    # reverse tensor to RGB image
    image = image.detach().cpu().numpy()#.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image

def visualize_heatmap(pred, gt):
    fig = plt.figure(num='heatmap', figsize=(8, 6))
    fig.clear()

    _format_heatmap(pred[0, 0], ax=plt.subplot(121))
    _format_heatmap(gt[0, 0], ax=plt.subplot(122))

    return fig
    
def visualize_bboxes(image, calibs, objects, preds):
    MAXCOLOR=40
    fig = plt.figure(num='bbox', figsize=(10, 8))
    fig.clear()

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    cmap = cm.get_cmap('tab20', MAXCOLOR)

    _format_bboxes(image, calibs, objects, cmap=cmap, ax=ax1)
    ax1.set_title('GroundTruth')

    _format_bboxes(image, calibs, preds, ax=ax2)
    ax2.set_title('Prediction')

    return fig


def _format_heatmap(heatmap, ax=None):
    heatmap = (heatmap.detach().cpu().numpy() * 255).astype(np.uint8)
    heatmap = grid_rot180(heatmap)

    # Create a new axis if one is not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    # Plot scores
    ax.clear()
    ax.imshow(heatmap)

    # Format axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax

def _format_bboxes(image, calib, objects, cmap=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()

    # Visualize image
    ax.imshow(visualize_image(image).transpose(1, 2, 0))
    extents = ax.axis()

    # Visualize objects
    if cmap is None:
        cmap = cm.get_cmap('tab20', len(objects))  

    for i, obj in enumerate(objects):
        ax = draw_3DBBox(ax, to_numpy(obj.dimension), to_numpy(obj.rotation), to_numpy(obj.location), to_numpy(calib), cmap(i), 2)

    ax.axis(extents)
    ax.axis(False)
    ax.grid(False)
    return ax