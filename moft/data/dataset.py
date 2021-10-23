import os, json, sys
sys.path.append(os.getcwd())

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor

from moft.data.multiviewX import MultiviewX
from moft.data.multiviewC import MultiviewC
from moft.utils import make_grid

class frameDataset(VisionDataset):
    def __init__(self, base:MultiviewC, transform = ToTensor(), 
                split='train', train_ratio = 0.9):
        super().__init__(base.root, transform=transform )
        assert split in ['train', 'val'], 'split mode error'
        # the unit of grid size and grid res is centimeter
        self.world_size, self.cube_LWH, self.reduced_grid_size = base.world_size, base.cube_LWH, base.reduced_grid_size
        self.base, self.root, self.num_cam, self.num_frame = base, base.root, base.num_cam, base.num_frame
        self.intrinsic_matrices, self.extrinsic_matrices= base.intrinsic_matrices, base.extrinsic_matrices

        if split == 'train':
            self.frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            self.frame_range = range(int(self.num_frame * train_ratio), self.num_frame)
        
        
        self.classAverage = base.classAverage
        self.labels, self.heatmaps = self.split(base.labels, base.heatmaps)
        self.fpaths = self.base.get_image_fpaths(self.frame_range)
        self.grid = make_grid(world_size=self.world_size, cube_LW=self.cube_LWH[:2]) # (l, w, 3)
    
    def split(self, labels, heatmaps):
        assert len(labels) == len(heatmaps), 'the number of labels must be equal to that of heatmaps'
        labels = [labels[i] for i in range(len(labels)) if i in self.frame_range]
        heatmaps = [heatmaps[i] for i in range(len(heatmaps)) if i in self.frame_range]
                
        return labels, heatmaps

    def __len__(self):
        return len(self.frame_range)
    
    def __getitem__(self, index: int):
        img_index = self.frame_range[index]
        batch_img_fpaths = [ self.fpaths[cam][img_index] for cam in range(1, self.num_cam + 1) ]
        images = [ self.transform(Image.open(p).convert('RGB')) for p in batch_img_fpaths ]
        calibs = [ self.intrinsic_matrices[cam] @ self.extrinsic_matrices[cam] for cam in range(self.num_cam)]
        objects = self.labels[index]
        heatmaps = torch.Tensor(self.heatmaps[index])
        grid = self.grid
        return index, images, objects, heatmaps, calibs, grid

if __name__ == '__main__':
    from moft.visualization.figure import _format_bboxes, _format_bottom
    import matplotlib.pyplot as plt
    from train import parse, mx_opts
    args = parse(mx_opts)
    # test MultiviewC and varify the visualization 
    # data = frameDataset(MultiviewC())
    # index, images, objects, heatmaps, calibs, grid = next(iter(data))
    # for cam in range(0, 7):
    #     ax = _format_bboxes(images[cam], calibs[cam], objects)
    #     plt.show()
    
    # test MultiviewX
    data = frameDataset(MultiviewX(root=r'F:\ANU\ENGN8602\Data\MultiviewX'))
    index, images, objects, heatmaps, calibs, grid  = next(iter(data))
    for cam in range(0, data.num_cam):
        ax = _format_bottom(images[cam], calibs[cam], objects, args)
        plt.show()
    pass
