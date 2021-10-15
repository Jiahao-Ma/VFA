import os, sys
from torch.nn.modules.module import Module

from tqdm.std import tqdm
sys.path.append(os.getcwd())
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import multivariate_normal

from moft.data.multiviewC import MultiviewC
from moft.data.dataset import Obj3D, frameDataset
from moft.data.smooth_label import gaussian_label


class ObjectEncoder(object):
    
    def __init__(self, dataset:frameDataset,
                     classname=['Cow'], 
                     map_sigma = 1.,
                     map_kernel_size = 10,
                     angle_range=360,
                     angle_radius=6,
                     topk=100):
        
        self.dataset = dataset
        self.classname = classname
        self.nclass = len(classname)
        # angle range and radius are the params of CSL
        self.angle_range=angle_range
        self.angle_radius = angle_radius
        self.topk = topk
        # world_size: (3900, 3900), cube_LW: (30, 30), cube_H: 32; units of all are centimeter(cm)
        self.world_size, self.cube_LW, self.cube_H = np.array(dataset.world_size), np.array(dataset.cube_LW), dataset.cube_H
        self.grid_size = list( map( lambda x: x // self.cube_LW, self.world_size) )
        self.map_kernel = self.gaussian_kernel(map_sigma, map_kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)

    def batch_encode(self, objects, heatmaps, grids):
        # Encode element by element
        batch_encoded = [self.encode(objs, heatmap, grid) \
                     for objs, heatmap, grid in zip(objects, heatmaps, grids)]
        return batch_encoded

    def encode(self, objects:Obj3D, heatmap:torch.Tensor, grid:torch.Tensor, visualize=False):
        # Filter the object by class name. MultiviewC only has on class
        objects = [obj for obj in objects if obj.classname in self.classname]

        # Return empty encode if there are no any objects
        if len(objects) == 0:
            return self._encode_empty(grid)

        # classids = torch.tensor([self.classname.index(obj.classname) for obj in objects],\
        #                         device=grid.device)
        location = grid.new([obj.location for obj in objects]) # [n, 3]
        dimension = grid.new([obj.dimension for obj in objects]) # [n, 3]
        rotation = grid.new([obj.rotation for obj in objects]) # [n, ]

        # Assign the target to the grid 
        # mask: dict key: 0 - nclass, value: torch.Tensor, storing the distribution of targets
        mask, indices = self._assign_to_grid(location, grid) 

        # Encode heatmap
        # heatmap = self._encode_heatmap(mask)
        heatmap = heatmap[None, None, :, :] 
        
        if visualize:
            viz_heatmaps = (heatmap.squeeze(0).squeeze(0) * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
            plt.imshow(viz_heatmaps)
            plt.show()
      
        # Encode location
        location_offset = self._encode_location(location, grid)

        # Encode dimension
        dimension_offset = self._encode_dimension(dimension, grid, indices)

        # Encode rotation
        rotation = self._encode_rotation(rotation, grid, indices)

        encoded_gt = {'mask' : mask, 
                   'heatmap' : heatmap,
                   'loc_offset' : location_offset.permute(0, 2, 3, 1),
                   'dim_offset' : dimension_offset.permute(0, 2, 3, 1),
                   'rotation' : rotation.permute(0, 2, 3, 1)}
        return encoded_gt

    def _encode_empty(self, grid):
        # if empty, encode mask(1, H, W), heatmaps(1, H, W), 
        # location_offsets(2, H, W), dimension_offsets(3, H, W), rotation(360, H, W)
        mask = grid.new_zeros(1, *grid.size()[:-1])
        heatmaps = grid.new_zeros(1, *grid.size()[:-1])
        location_offsets = grid.new_zeros(2, *grid.size()[:-1])
        dimension_offsets = grid.new_zeros(3, *grid.size()[:-1])
        rotation = grid.new_zeros(self.angle_range, *grid.size()[:-1])
        return mask, heatmaps, location_offsets, dimension_offsets, rotation

    def _assign_to_grid(self, location, grid):
        location = location[..., :2]
        # normalize locations
        location = location / location.new(self.world_size).view(-1, 2) * grid.size()[0]
        foreground = grid.new_zeros(1, *grid.size()[:-1]) # B, 1, H, W
        indices = list()
        for loc in location:
            coord_x, coord_y = int(loc[0]), int(loc[1])
            foreground[:, coord_y, coord_x] = 1.
            indices.append([coord_x, coord_y])
        return foreground.unsqueeze(0), indices

    def _encode_heatmap(self, mask):
        heatmap = mask.to(torch.float32)  # (1, 1, H, W)
        with torch.no_grad():
            heatmap = F.conv2d(heatmap, self.map_kernel.float().to(heatmap.device), padding=int((self.map_kernel.shape[-1] - 1) / 2))
        return heatmap.clip(min=0., max=1.)

    def _encode_location(self, location, grid):
        # z coordinate value of target is zero by default
        # thus, location offset is (2, H, W)
        location = location[..., :2]
        location = location / location.new(self.world_size).view(-1, 2) * grid.size()[0] # normalize location
        location_offset = grid.new_zeros((1, 2, *grid.size()[:-1]))
        for loc in location:
            coord_x, coord_y = int(loc[0]), int(loc[1])
            offset_x = loc[0] - coord_x
            offset_y = loc[1] - coord_y
            location_offset[:, 0, coord_y, coord_x] = offset_x
            location_offset[:, 1, coord_y, coord_x] = offset_y

        return location_offset
        
    def _encode_dimension(self, dimension, grid, indices):
        # default one class 'Cow'
        dimension_mean = self.dataset.classAverage.get_mean(self.classname[0])
        dimension_mean = dimension.new(dimension_mean) # convert to same device and dtype
        # FORMULA: exp(dim_off) * dim_mean = dim
        dimension_offset = grid.new_zeros((1, 3, *grid.size()[:-1])) # (3, H, W)
        for dim, index in zip(dimension, indices):
            coord_x, coord_y = index
            # offset of height, width, length
            offset_dim = torch.log(dim / dimension_mean) 
            dimension_offset[:, 0, coord_y, coord_x] = offset_dim[0] # offset_height
            dimension_offset[:, 1, coord_y, coord_x] = offset_dim[1] # offset_width
            dimension_offset[:, 2, coord_y, coord_x] = offset_dim[2] # offset_length

        return dimension_offset

    def _encode_rotation(self, rotation, grid, indices):
        # angle => [cos(angle), sin(angle)] discarded.
        # TODO: CSL for angle prediction
        rotation_offset = grid.new_zeros((1, *grid.size()[:-1], self.angle_range)) #(360, H, W)
        for angle, index in zip(rotation, indices):
            coord_x, coord_y = index
            smooth_label = gaussian_label(torch.rad2deg(angle).item(), self.angle_range, sigma=self.angle_radius)
            rotation_offset[:, coord_y, coord_x, :] = torch.tensor(smooth_label)[None, :]
        
        return rotation_offset.permute(0, 3, 1, 2)

    def gaussian_kernel(self, map_sigma = 1., map_kernel_size = 10):
        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        kernel = kernel / kernel.max()
        kernel_size = kernel.shape[0]
        map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        map_kernel[0, 0] = torch.from_numpy(kernel)
        return map_kernel
    
    def nms(self, heatmap):
        mask = torch.eq(self.maxpool(heatmap), heatmap).to(heatmap.dtype)
        return mask * heatmap

    def decode(self, pred, cls_thresh):
        heatmap, tytx, thtwtl, orient = pred['heatmap'],  pred['loc_offset'],\
                                        pred['dim_offset'], pred['rotation']
        device, dtype = heatmap.device, heatmap.dtype                                    
        heatmap = self.nms(torch.sigmoid(heatmap))
        # (1, 1, L, W)
        heatmap = heatmap.flatten(start_dim=2).transpose(1, 2)
        heatmap_conf, _ = torch.max(heatmap, dim=-1)
        L, W = pred['heatmap'].shape[2:]
        grid_y, grid_x = torch.meshgrid( torch.arange(L, dtype=dtype, device=device), \
                                         torch.arange(W, dtype=dtype, device=device))
        # Decode location
        tytx = torch.sigmoid(tytx)
        bboxes_cy = (grid_y[None, ...] + tytx[..., 0]).flatten(start_dim=1) / self.grid_size[0] * self.world_size[0]
        bboxes_cx = (grid_x[None, ...] + tytx[..., 1]).flatten(start_dim=1) / self.grid_size[1] * self.world_size[1]
        # Decode dimension
        dimension_mean = self.dataset.classAverage.get_mean(self.classname[0])
        bboxes_h = torch.exp(thtwtl[..., 0]).flatten(start_dim=1) * dimension_mean[0]
        bboxes_w = torch.exp(thtwtl[..., 1]).flatten(start_dim=1) * dimension_mean[1]
        bboxes_l = torch.exp(thtwtl[..., 2]).flatten(start_dim=1) * dimension_mean[2]
        # Decode rotation
        orient = torch.sigmoid(orient)
        _, orient_idx = torch.max(orient, dim=-1)
        orient_idx = orient_idx.flatten(start_dim=1)
        # Concatenate conf, location, dimension and rotation
        _, topk_index = torch.topk(heatmap_conf, k=self.topk, dim=1)
        # output: list contain tensor [1, topk]
        output = [ torch.gather(x, dim=1, index=topk_index)
                   for x in [heatmap_conf, bboxes_cy, bboxes_cx, bboxes_h, bboxes_w, bboxes_l, orient_idx] ]
        # Construct output
        mask = output[0] > cls_thresh
        conf = output[0][mask]
        location = torch.stack([output[2][mask], output[1][mask], torch.zeros_like(output[1][mask])], dim=-1) # x y z
        dimension = torch.stack([output[3][mask], output[4][mask], output[5][mask]], dim=-1)
        rotation = torch.deg2rad(output[6][mask])
        return {'conf': conf,
                'location': location,
                'dimension': dimension,
                'rotation': rotation
                }
    def batch_decode(self, pred, cls_thresh):
        batch = self.decode(pred, cls_thresh)
        objects = list()
        for i in range(len(batch['conf'])):
            objects.append(Obj3D(
                  classname='Cow',
                  conf=batch['conf'][i],
                  location=batch['location'][i],
                  dimension=batch['dimension'][i],
                  rotation=batch['rotation'][i]))
        return objects

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from moft.utils import collate
    from moft.model.oftnet import MOFTNet

    dataset = frameDataset(MultiviewC())
    encoder = ObjectEncoder(dataset)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate)

    index, images, objects, calibs, grid = next(iter(dataloader))
    batch_encoded = encoder.batch_encode(objects, grid)
    model = MOFTNet()
    encode_pred = model(images, calibs, grid)
    output = encoder.decode(encode_pred, 0.5)
    pass
