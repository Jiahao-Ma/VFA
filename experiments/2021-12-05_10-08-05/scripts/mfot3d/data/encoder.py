import os, sys
sys.path.append(os.getcwd())
from torch.nn.modules.module import Module

from tqdm.std import tqdm


sys.path.append(os.getcwd())
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import multivariate_normal

from vfa.utils import Obj2D, Obj3D
from vfa.data.multiviewC import MultiviewC
from vfa.data.multiviewX import MultiviewX
from vfa.data.wildtrack import Wildtrack
from vfa.data.dataset import frameDataset
from vfa.data.smooth_label import gaussian_label


class ObjectEncoder(object):
    
    def __init__(self, dataset:frameDataset,
                     map_sigma = 1.,
                     map_kernel_size = 10,
                     angle_range=360,
                     angle_radius=6,
                     topk=100, 
                     kernel_type='RGK'):
        
        self.dataset = dataset
        self.classname = dataset.base.label_names
        self.nclass = len(self.classname)
        # angle range and radius are the params of CSL
        self.angle_range=angle_range
        self.angle_radius = angle_radius
        self.topk = topk
        # MultiviewC: world_size: (3900, 3900), cube_LWH: (30, 30, 32); units of all are centimeter(cm)
        # MultiviewX: world_size: (640, 1000), real_world_size:(16m, 25m), cube_LWH:(4, 4, 36)
        # Wildtrack: world: (480, 1440) real_world_size:(12m, 36m)
        self.world_size, self.cube_LWH = np.array(dataset.world_size), np.array(dataset.cube_LWH)
        self.grid_size = self.world_size / self.cube_LWH[:2]
        if kernel_type == 'GK' and self.dataset.base.__name__ == 'MultiviewC':
            self.map_kernel = self.gaussian_kernel(map_sigma, map_kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)

    def batch_encode(self, objects, heatmaps, grids):
        if self.dataset.base.__name__ in ['MultiviewC', 'MVM3D']:
            # Encode element by element
            batch_encoded = [self.encode3d(objs, heatmap, grid) \
                        for objs, heatmap, grid in zip(objects, heatmaps, grids)]
            return batch_encoded
        elif self.dataset.base.__name__ in ['MultiviewX', 'Wildtrack']:
            batch_encoded = [self.encode2d(objs, heatmap, grid) \
                        for objs, heatmap, grid in zip(objects, heatmaps, grids)]
            return batch_encoded
        else:
            raise ValueError("""Dataset Error: only support `MultivewC` `MVM3D` for 3D detection, 
                                and `MultiviewX` `Wildtrack` for 2D detection.""")

    def encode3d(self, objects:Obj3D, heatmap:torch.Tensor, grid:torch.Tensor, visualize=False):
        # Filter the object by class name. MultiviewC only has on class
        objects = [obj for obj in objects if obj.classname in self.classname]

        # Return empty encode if there are no any objects
        if len(objects) == 0:
            return self._encode_empty3d(grid)

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

    def encode2d(self, objects:Obj2D, heatmap:torch.Tensor, grid:torch.Tensor, visualize=False):
        # Filter the object by class name. MultiviewX, WildTrack only has on class
        objects = [obj for obj in objects if obj.classname in self.classname]

        # Return empty encode if there are no any objects
        if len(objects) == 0:
            return self._encode_empty2d(grid)
   
        location = grid.new([obj.location for obj in objects]) # [n, 3]

        # Assign the target to the grid
        mask, _ = self._assign_to_grid(location, grid)

        if visualize:
            viz_heatmaps = (heatmap * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
            plt.imshow(viz_heatmaps)
            plt.show()

        # Encode heatmap
        # heatmap = self._encode_heatmap(mask)
        heatmap = heatmap[None, None, :, :] 

        # Encode location
        location_offset = self._encode_location(location, grid)

        encoded_gt = {'mask' : mask, 
                      'heatmap' : heatmap,
                      'loc_offset' : location_offset.permute(0, 2, 3, 1)}
        return encoded_gt
        
        
    def _encode_empty3d(self, grid):
        # if empty, encode mask(1, H, W), heatmaps(1, H, W), 
        # location_offsets(2, H, W), dimension_offsets(3, H, W), rotation(360, H, W)
        mask = grid.new_zeros(1, *grid.size()[:-1])
        heatmaps = grid.new_zeros(1, *grid.size()[:-1])
        location_offsets = grid.new_zeros(2, *grid.size()[:-1])
        dimension_offsets = grid.new_zeros(3, *grid.size()[:-1])
        rotation = grid.new_zeros(self.angle_range, *grid.size()[:-1])
        return mask, heatmaps, location_offsets, dimension_offsets, rotation

    def _encode_empty2d(self, grid):
        mask = grid.new_zeros(1, *grid.size()[:-1])
        heatmaps = grid.new_zeros(1, *grid.size()[:-1])
        location_offsets = grid.new_zeros(2, *grid.size()[:-1])
        return mask, heatmaps, location_offsets

    def _assign_to_grid(self, location, grid):
        location = location[..., :2]
        # normalize locations
        location = location / location.new(self.world_size).view(-1, 2) * location.new([grid.size()[:2]]) 
        foreground = grid.new_zeros(1, *grid.size()[:-1]) # B, 1, H, W
        indices = list()
        for loc in location:
            coord_x, coord_y = int(loc[0]), int(loc[1])
            if self.dataset.base.__name__ == Wildtrack.__name__:
                foreground[:, coord_x, coord_y] = 1.
            else:
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
        location = location / location.new(self.world_size).view(-1, 2) * location.new([grid.size()[:2]])# normalize location
        location_offset = grid.new_zeros((1, 2, *grid.size()[:-1]))
        for loc in location:
            coord_x, coord_y = int(loc[0]), int(loc[1])
            offset_x = loc[0] - coord_x
            offset_y = loc[1] - coord_y
            if self.dataset.base.__name__ == Wildtrack.__name__:
                location_offset[:, 0, coord_x, coord_y] = offset_x
                location_offset[:, 1, coord_x, coord_y] = offset_y
            else:
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

    def decode3d(self, pred, cls_thresh):
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
        rotation = torch.deg2rad(output[6][mask].to(torch.float32))
        return {'conf': conf,
                'location': location,
                'dimension': dimension,
                'rotation': rotation
                }
            
    def decode2d(self, pred, cls_thresh):
        heatmap, tytx = pred['heatmap'],  pred['loc_offset']
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
        
        _, topk_index = torch.topk(heatmap_conf, k=self.topk, dim=1)
        # output: list contain tensor [1, topk]
        output = [ torch.gather(x, dim=1, index=topk_index)
                   for x in [heatmap_conf, bboxes_cy, bboxes_cx] ] # TODO: check bboxes_cx, bboxes_cy ?
        # Construct output
        mask = output[0] > cls_thresh
        conf = output[0][mask]
       
        if self.dataset.base.__name__ == 'Wildtrack':
            location = torch.stack([output[1][mask], output[2][mask], torch.zeros_like(output[1][mask])], dim=-1) # x y z
        else:
            location = torch.stack([output[2][mask], output[1][mask], torch.zeros_like(output[1][mask])], dim=-1) # x y z
        
        return {'conf': conf,
                'location': location
                }                           
    
    def batch_decode(self, pred, cls_thresh):
        # for MultiviewC, MVM3D dataset
        if self.dataset.base.__name__ in ['MultiviewC', 'MVM3D']:
            batch = self.decode3d(pred, cls_thresh)
            objects = list()
            for i in range(len(batch['conf'])):
                objects.append(Obj3D(
                    classname=self.dataset.base.label_names[0], # default one class
                    conf=batch['conf'][i],
                    location=batch['location'][i],
                    dimension=batch['dimension'][i],
                    rotation=batch['rotation'][i]))
            return objects
        # for MultiviewX, WildTrack dataset
        elif self.dataset.base.__name__ in ['MultiviewX', 'Wildtrack']:
            batch = self.decode2d(pred, cls_thresh)
            objects = list()
            for i in range(len(batch['conf'])):
                objects.append(Obj2D(
                    classname=self.dataset.base.label_names[0],
                    conf=batch['conf'][i],
                    location=batch['location'][i],
                    ))
            return objects
        else:
            raise ValueError("""Dataset Error: only support `MultivewC` `MVM3D` for 3D detection, 
                                and `MultiviewX` `Wildtrack` for 2D detection.""")

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from vfa.utils import collate
    from vfa.model.vfanet import VFANet
    from vfa.config import mc_opts
    # verify 3d
    # dataset = frameDataset(MultiviewC(root=r'F:\ANU\ENGN8602\Data\MultiviewC_github\dataset'))
    # encoder = ObjectEncoder(dataset)
    # dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate)

    # index, images, objects, heatmaps, calibs, grid = next(iter(dataloader))
    # batch_encoded = encoder.batch_encode(objects, heatmaps, grid)
    # model = VFANet(mc_opts)
    # encode_pred = model(images, calibs, grid)
    # output = encoder.batch_decode(encode_pred, 0.5)
    
    # verify 2d
    dataset = frameDataset(MultiviewX(root=r'F:\ANU\ENGN8602\Data\MultiviewX'))
    encoder = ObjectEncoder(dataset)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate)

    index, images, objects, heatmaps, calibs, grid = next(iter(dataloader))
    batch_encoded = encoder.batch_encode(objects, heatmaps, grid)
    model = VFANet(mc_opts,
                    grid_height=180,
                    cube_size=(4, 4, 36),
                    mode='2D')
    encode_pred = model(images, calibs, grid)
    output = encoder.batch_decode(encode_pred, 0.5)
    pass
