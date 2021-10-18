import os, json, sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor
from collections import namedtuple

from moft.data.RGK import RotationGaussianKernel
from moft.data.multiviewC import MultiviewC
from moft.data.ClsAvg import ClassAverage
from moft.utils import make_grid

Obj3D = namedtuple('Obj3D',
        ['classname', 'dimension', 'location', 'rotation', 'conf'])

class frameDataset(VisionDataset):
    def __init__(self, base:MultiviewC, transform = ToTensor(), world_size=(3900, 3900),
                  cube_LW=25, cube_H=32, split='train', train_ratio = 0.9, reload_RGK=False):
        super().__init__(base.root, transform=transform )
        assert split in ['train', 'val'], 'split mode error'
        # the unit of grid size and grid res is centimeter
        self.world_size, self.cube_LW, self.cube_H = world_size, cube_LW, cube_H
        self.base, self.root, self.num_cam, self.num_frame = base, base.root, base.num_cam, base.num_frame
        self.intrinsic_matrices, self.extrinsic_matrices, self.R_z = base.intrinsic_matrices, base.extrinsic_matrices, base.R_z

        if split == 'train':
            self.frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            self.frame_range = range(int(self.num_frame * train_ratio), self.num_frame)
        
        self.RGK = RotationGaussianKernel()
        self.reload_RGK=reload_RGK
        self.reduced_grid_size = list( map( lambda x: int(x / cube_LW), self.world_size ) )
        self.classAverage = ClassAverage(classes=['Cow'])
        self.labels, self.heatmaps = self.download()
        self.fpaths = self.base.get_image_fpaths(self.frame_range)
        self.grid = make_grid(world_size=world_size, cube_LW=cube_LW) # (l, w, 3)
    
    def download(self):
        ann_paths = [ os.path.join(self.base.ann_root, p) for p in os.listdir(self.base.ann_root) ]
        labels = list()
        # if cls avg not exist (true), calculate the property of dataset, including mean, number and sum of dimension
        BuildClsAvg = not os.path.exists(self.classAverage.save_path) 
        # if RGK not exist (true), build RGK; else, load RGK from file
        BuildRGK = self.reload_RGK or not self.RGK.RGKExist() 
        with tqdm(total=len(ann_paths), postfix=dict, mininterval=0.3) as pbar:
            for i, ann_path in enumerate(ann_paths):
                with open(ann_path, 'r') as f:
                    annotations = json.load(f)
                cow_infos = list()
                heatmap = np.zeros(self.reduced_grid_size, dtype=np.float32)
                for cows in annotations['C1']:
                    location = cows['location']
                    dimension = cows['dimension']
                    rotation = np.deg2rad(cows['rotation']) # -180~180 => -pi~pi 
                    cow_infos.append(Obj3D(classname='Cow', dimension=dimension, 
                                        location=location, rotation=rotation, conf=None))
                    if BuildRGK:
                        x, y, _ = location
                        _, w, l = dimension
                        box_cx = x * self.reduced_grid_size[0] / self.world_size[0]
                        box_cy = y * self.reduced_grid_size[1] / self.world_size[1]
                        heatmap = self.RGK.gaussian_kernel_heatmap(heatmap, box_cx, box_cy, l, w, cows['rotation'])
                    if BuildClsAvg:
                        self.classAverage.add_item('Cow', dimension)
                if BuildRGK:
                    self.RGK.add_item(heatmap)
                labels.append(cow_infos)
                pbar.set_postfix(**{ 'Process' : ' {} / {}'.format(i, len(ann_paths)),
                                    'Fname' : ' {}'.format(os.path.basename(ann_path)) })
                pbar.update(1)

        if BuildClsAvg:
            self.classAverage.dump_to_file()
        else:
            self.classAverage.load_from_file()
        if BuildRGK:
            # dump RGK to file
            heatmaps = self.RGK.dump_to_file()
        else:
            heatmaps = self.RGK.load_from_file()

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
        return index, images, objects, heatmaps, calibs, self.grid

if __name__ == '__main__':
    # Test the dataset
    from moft.visualization.figure import _format_bboxes
    import matplotlib.pyplot as plt
    data = frameDataset(MultiviewC(root=r'F:\ANU\ENGN8602\Data\MultiviewC_github\dataset')) # The Path of MultiviewC dataset
    index, images, objects, heatmaps, calibs, grid = next(iter(data))
    for cam in range(0, 7):
        ax = _format_bboxes(images[cam], calibs[cam], objects)
        plt.show()
    


