import os, sys
sys.path.append(os.getcwd())
import numpy as np
from torch.utils.data import DataLoader
from moft.utils import collate
from moft.data.dataset import frameDataset, MultiviewC
from moft.data.encoder import ObjectEncoder
import matplotlib.pyplot as plt
from PIL import Image
def homopraphy_project(imgcoord, proj):
    # project image coord to world coord by homography transformation
    # imgcoord: (n, 2) worldcoord: (n, 3)
    imgcoord = imgcoord.T
    imgcoord = np.concatenate([imgcoord, np.ones(shape=(1, imgcoord.shape[1]))], axis=0)
    proj = np.linalg.inv(np.delete(proj, 2, 1))
    worldcoord = proj @ imgcoord
    return (worldcoord[:2, :] / worldcoord[2:, :]).T

def grid_rot180(arr):
    arr = arr[::-1, :]
    arr = arr[:, ::-1]
    return arr


dataset = frameDataset(MultiviewC())

index, images, objects, calibs, grid = next(iter(dataset))
imgHW = (720, 1280)
INTERVAL = 1
GRIDH, GRIDW = 3900, 3900
NGRIDH, NGRIDW = 130, 130
xx, yy = np.meshgrid(np.arange(0, imgHW[1], INTERVAL), np.arange(0, imgHW[0], INTERVAL))
imgcoord = np.stack([xx, yy], axis=2).reshape(-1, 2)
for cam in range(dataset.num_cam):
    world_grid_map = np.zeros((NGRIDH, NGRIDW))
    calib = calibs[cam]
    image = images[cam]
    worldcoord = homopraphy_project(imgcoord, calib)
    insides = ( worldcoord[:, 0] >=0 ) * (worldcoord[:, 0] <= GRIDH ) * ( worldcoord[:, 1] >=0 ) * (worldcoord[:, 1] <= GRIDW )
    worldcoord = worldcoord[insides]
    # normalize world coord, resize the scale of [0, 3900] to [0, 130]
    worldcoord = worldcoord / np.array([GRIDH, GRIDW]) * np.array([NGRIDH, NGRIDW])
    for coord in worldcoord:
        world_grid_map[int(coord[1]), int(coord[0])] += 1
    world_grid_map = world_grid_map.astype(np.uint8)
    world_grid_map = grid_rot180(world_grid_map)
    world_grid_map = Image.fromarray(world_grid_map)
    plt.imshow(world_grid_map)
    plt.show()