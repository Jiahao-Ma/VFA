import os, json, sys

from torch.nn.functional import hardtanh
sys.path.append(os.getcwd())

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor

from moft.data.multiviewX import MultiviewX
from moft.data.multiviewC import MultiviewC
from moft.data.wildtrack import Wildtrack
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
            if base.__name__ == Wildtrack.__name__:
                self.frame_range = range(0, int(self.num_frame * train_ratio), 5)
            else:
                self.frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            if base.__name__ == Wildtrack.__name__:
                self.frame_range = range(int(self.num_frame * train_ratio), self.num_frame, 5)
            else:
                self.frame_range = range(int(self.num_frame * train_ratio), self.num_frame)
        
        
        self.classAverage = base.classAverage
        self.labels, self.heatmaps = self.split(base.labels, base.heatmaps)
        self.fpaths = self.base.get_image_fpaths(self.frame_range)
        self.grid = make_grid(world_size=self.world_size, cube_LW=self.cube_LWH[:2], dataset=base.__name__) # (l, w, 3)
        pass
    
    def split(self, labels, heatmaps):
        assert len(labels) == len(heatmaps), 'the number of labels must be equal to that of heatmaps'
        if self.base.__name__ == Wildtrack.__name__:
            labels = [labels[id] for id, i in enumerate(range(0, self.num_frame, 5)) if i in self.frame_range]
            heatmaps = [heatmaps[id] for id, i in enumerate(range(0, self.num_frame, 5)) if i in self.frame_range]
        else:
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
    from moft.visualization.bbox import project
    import matplotlib.pyplot as plt
    from train import parse, mx_opts, wt_opts
    wt_opts.cube_size=np.array((40, 40, 8))
    args = parse(wt_opts)
    # test MultiviewC and varify the visualization 
    # data = frameDataset(MultiviewC())
    # index, images, objects, heatmaps, calibs, grid = next(iter(data))
    # for cam in range(0, 7):
    #     ax = _format_bboxes(images[cam], calibs[cam], objects)
    #     plt.show()
    
    # test MultiviewX
    # data = frameDataset(MultiviewX(root=r'F:\ANU\ENGN8602\Data\MultiviewX'))
    # index, images, objects, heatmaps, calibs, grid  = next(iter(data))
    # for cam in range(0, data.num_cam):
    #     ax = _format_bottom(images[cam], calibs[cam], objects, args)
    #     plt.show()

    # test MultiviewX
    r = 1
    data = frameDataset(Wildtrack(root=args.root,
                                  cube_LWH=args.cube_size))
    index, images, objects, heatmaps, calibs, grid  = next(iter(data))
    colors = ['green', 'purple']
    heatmaps = (heatmaps.cpu().numpy()*255).clip(0, 255)
    for cam in range(0, data.num_cam):
        fig, axes = plt.subplots(1, 2)
        axes[0] = _format_bottom(images[cam], calibs[cam], objects, args, ax=axes[0], height=16)
        
        bottom = grid.cpu().numpy().reshape(-1, 3).transpose()
        bottom = Wildtrack.get_worldcoord_from_worldgrid(bottom).transpose() # N, 3
        bottom = np.concatenate([bottom, np.ones((bottom.shape[0], 1))], axis=1)
        imgcoord = project(bottom, calibs[cam])
        mask = (imgcoord[:, 0]>0) * (imgcoord[:, 0]<1920) * (imgcoord[:, 1]>0) * (imgcoord[:, 1]<1080)
        axes[0].scatter(imgcoord[:, 0], imgcoord[:, 1], s=5, c='green')

        vis_mask = grid.cpu().numpy().reshape(-1, 3)[mask]
        vis_mask = (vis_mask / np.array([4, 4, 1]))[:, :2].astype(np.int32)
        temp = heatmaps.copy()
        for pts in vis_mask:
            temp[pts[0]-r:pts[0]+r, pts[1]-r:pts[1]+r] += 150
        axes[1].imshow(temp.clip(0, 255))
        plt.show()
    pass
