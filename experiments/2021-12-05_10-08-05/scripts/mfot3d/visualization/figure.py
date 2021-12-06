import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from vfa.utils import to_numpy, grid_rot180
from vfa.data.multiviewX import MultiviewX
from vfa.data.wildtrack import Wildtrack
from .bbox import draw_3DBBox, project
def visualize_image(image):
    # image format: (3, H, W) value range: (0, 1)
    # reverse tensor to RGB image
    image = image.detach().cpu().numpy()#.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image

def visualize_heatmap(pred, gt):
    fig = plt.figure(num='heatmap', figsize=(8, 6))
    fig.clear()

    _format_heatmap(pred[0, 0], ax=plt.subplot(211))
    _format_heatmap(gt[0, 0], ax=plt.subplot(212))

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

def visualize_bottom(image, calibs, objects, preds, args):

    fig = plt.figure(num='bbox', figsize=(10, 8))
    fig.clear()

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    _format_bottom(image, calibs, objects, args, ax=ax1)
    ax1.set_title('GroundTruth')

    _format_bottom(image, calibs, preds, args, ax=ax2)
    ax2.set_title('Prediction')

    return fig

def _format_bottom(image, calib, objects, args, ax=None, height=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()

    # Visualize image
    image = Image.fromarray(visualize_image(image).transpose(1, 2, 0))
    image = image.resize(args.image_size[::-1])
    ax.imshow(image)
    extents = ax.axis()
    ax.axis(extents)
    ax.axis(False)
    ax.grid(False)

    # Construct homography coord of bottom
    bottom = list()
    head = list()
    for obj in objects:
        worldcoord = np.zeros((3), dtype=np.float32)
        if args.data == MultiviewX.__name__:
            worldcoord[:2] = MultiviewX.get_worldcoord_from_worldgrid(to_numpy(obj.location[:2]))
        elif args.data == Wildtrack.__name__:
            worldcoord[:2] = Wildtrack.get_worldcoord_from_worldgrid(to_numpy(obj.location[:2]))
            if height is not None:
                obj.location[2] = height
                headcoord = Wildtrack.get_worldcoord_from_worldgrid(to_numpy(obj.location))
        else:
            raise ValueError('Unknow dataset. Expect {} and {}, but got{}.'\
                            .format(MultiviewX.__name__, 
                                    Wildtrack.__name__,
                                    args.data
                                    ))
        bottom.append(worldcoord)
        if height is not None:
            head.append(headcoord)

    bottom = np.array(bottom).reshape(-1, 3)
    bottom3d = np.concatenate([bottom, np.ones((bottom.shape[0], 1))], axis=1)
    bottom2d = project(bottom3d, to_numpy(calib))
    # Visualize bottom center 
    ax.scatter(bottom2d[:, 0], bottom2d[:, 1], s=5, c='red')

    if height is not None:
        head = np.array(head).reshape(-1, 3)
        head3d = np.concatenate([head, np.ones((head.shape[0], 1))], axis=1)
        head2d = project(head3d, to_numpy(calib))
        # Visualize bottom center 
        ax.scatter(head2d[:, 0], head2d[:, 1], s=5, c='yellow')
    return ax