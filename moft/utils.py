import torch
import os, sys
import numpy as np
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import namedtuple
from collections import defaultdict

# for MultiviewC, MVM3D dataset
Obj3D = namedtuple('Obj3D',
        ['classname', 'dimension', 'location', 'rotation', 'conf'])
# for MultiviewX, WildTrack dataset
Obj2D = namedtuple('Obj2D', 
        ['classname', 'location', 'conf'])

def make_grid(world_size=(3900, 3900), grid_offset=(0, 0, 0), cube_LW=[25, 25], dataset='Wildtrack'):
    """
        *********
        *       *
        *       * y
        *********
                x
    """
    if dataset == 'Wildtrack':
        length, width = world_size[::-1]
    else:
        length, width = world_size
        
    xoff, yoff, zoff = grid_offset
    xcoords = torch.arange(0., width, cube_LW[0]) + xoff
    ycoords = torch.arange(0., length, cube_LW[1]) + yoff
    
    if dataset == 'Wildtrack':
        xx, yy = torch.meshgrid(xcoords, ycoords)
    else:
        yy, xx = torch.meshgrid(ycoords, xcoords)
    return torch.stack([xx, yy, torch.full_like(xx, zoff)], dim=-1)

def collate(batch):
    index, images, objects, heatmaps, calibs, grid = zip(*batch)

    index = torch.LongTensor(index)
    images = torch.stack([image for img_batch in images for image in img_batch])
    calibs = torch.stack([torch.Tensor(calib) for batch_calib in calibs for calib in batch_calib])
    grid = torch.stack(grid)
    heatmaps = torch.stack(heatmaps)

    return index, images, objects, heatmaps, calibs, grid

def project(vectors, calib):
    """
        Project points in 3D spaces to 2D planes 
    """
    # vectors: (1, n, L, W, 8, 3) n: the number of layer of grid map, L & W: the length and width of grid
    # calib: (1, 1, 1, 1, 1, 3, 4)
    vectors = vectors.unsqueeze(-1) #(1, 5, 130, 130, 8, 3, 1)
    homography = torch.matmul(calib[..., :-1], vectors) + calib[..., -1:] # (1, 5, 130, 130, 8, 3, 1)
    homography = homography.squeeze(-1)
    return homography[..., :-1] / homography[..., -1:]

class MetricDict(defaultdict):
    def __init__(self):
        super().__init__(float)
        self.count = defaultdict(int)
    
    def __add__(self, other):
        for key, value in other.items():
            self[key] += value
            self.count[key] += 1
        return self
    @property
    def mean(self):
        return { key: self[key] / self.count[key] for key in self.keys() }

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)

def grid_rot180(arr):
    if len(arr.shape) == 2:
        arr = arr[::-1, :]
        arr = arr[:, ::-1]
    elif len(arr.shape) == 3:
        arr = arr[:, ::-1, :]
        arr = arr[:, :, ::-1]
    return arr

def record(save_dir, content):
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))
        
    with open(save_dir, encoding='utf-8', mode='a') as f:
        f.write(content)
