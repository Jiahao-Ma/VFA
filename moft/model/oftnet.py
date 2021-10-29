import torch
import torch.nn as nn
import os, sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F

from moft.model.oft import OFT
import moft.model.resnet as resnet
from moft.data.multiviewX import MultiviewX

class MOFTNet(nn.Module):
    def __init__(self, args,
                 base='resnet18',
                 grid_height=160,
                 cube_size=(25, 25, 32),
                 angle_range=360,
                 mode='3D',
                 pretrained=False):
        super(MOFTNet, self).__init__()
        assert base in ['resnet18', 'resnet34'], 'Unrecognized model, expect `resnet18` or `resnet34`, got {}.'.format(base)
        assert mode in ['2D', '3D'], 'mode error, expect `2D` or `3D`, got{}'.format(mode)
 
        self.mode = mode
        resnet_model = getattr(resnet, base)(pretrained=pretrained)
        self.base = resnet_model

        self.oft8 = OFT(channel=256, grid_height=grid_height, cube_size=cube_size, feat_scale= 1 / 8., args=args)
        self.oft16 = OFT(channel=256, grid_height=grid_height, cube_size=cube_size, feat_scale= 1 / 16., args=args)
        self.oft32 = OFT(channel=256, grid_height=grid_height, cube_size=cube_size, feat_scale= 1 / 32., args=args)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]))

        self.lat8 = nn.Conv2d(128, 256, 1)
        self.lat16 = nn.Conv2d(256, 256, 1)
        self.lat32 = nn.Conv2d(512, 256, 1)

        self.bn8 = nn.GroupNorm(16, 256)
        self.bn16 = nn.GroupNorm(16, 256)
        self.bn32 = nn.GroupNorm(16, 256)

        self.fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                                  nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(256), nn.ReLU(True))
        # Detection head
        self.map_classifier = nn.Sequential( nn.Conv2d(256, 1, kernel_size=3, padding=4, dilation=4, bias=False) )
        self.tytx_pred = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(16, 256), nn.ReLU(True),
                                       nn.Conv2d(256, 2, kernel_size=3, padding=1, bias=False))
        if self.mode == '3D':
            self.orient_pred = nn.Sequential( nn.Conv2d(256, angle_range, kernel_size=3, padding=4, dilation=4, bias=False) )
            self.thtwtl_pred = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(16, 256), nn.ReLU(True),
                                        nn.Conv2d(256, 3, kernel_size=3, padding=1, bias=False))
    
    def forward(self, images, calibs, grid, visualize=False, visualize_ortho=False):
        # Normalize Image 
        # image size: (7, 3, iH, iW), calibs: (7, 3, 4), grid: (1, 156, 156, 3)
        images = (images - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        N, C, iH, iW = images.shape
        # feature :(7, 512, 90, 160)
        feats8, feats16, feats32 = self.base(images)
        
        ortho = 0
        for cam in range(N):
         
            calib = calibs[cam]
            feat8 = feats8[[cam], ...]
            feat16 = feats16[[cam], ...]
            feat32 = feats32[[cam], ...]

            lat8 = F.relu(self.bn8(self.lat8(feat8)))
            lat16 = F.relu(self.bn16(self.lat16(feat16)))
            lat32 = F.relu(self.bn32(self.lat32(feat32)))

            ortho_feat8 = self.oft8(lat8, calib, grid, (-1, 0.95), visualize_ortho)
            ortho_feat16 = self.oft16(lat16, calib, grid, (-1, 0.95),visualize_ortho)
            ortho_feat32 = self.oft32(lat32, calib, grid, (-1, 0.95), visualize_ortho)
            ortho_feats = ortho_feat8 + ortho_feat16 + ortho_feat32
        
            # Sum all ortho_feats up
            ortho += ortho_feats

            if visualize:
                fig = plt.figure(figsize=(15, 8))
                gs = gridspec.GridSpec(1, 2)
                gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0])
                gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])

                fig.add_subplot(gs00[0])
                viz_feature = torch.norm(feat8, dim=1)
                viz_feature = (viz_feature).detach().cpu().numpy()[0]
                plt.title('C%d Feat8 (90, 160)'%(cam+1))
                plt.axis('off')
                plt.imshow(viz_feature)

                fig.add_subplot(gs00[1])
                viz_feature = torch.norm(feat16, dim=1)
                viz_feature = (viz_feature).detach().cpu().numpy()[0]
                plt.title('C%d Feat16 (45, 80)'%(cam+1))
                plt.axis('off')
                plt.imshow(viz_feature)

                fig.add_subplot(gs00[2])
                viz_feature = torch.norm(feat32, dim=1)
                viz_feature = (viz_feature).detach().cpu().numpy()[0]
                plt.title('C%d Feat32 (23, 40)'%(cam+1))
                plt.axis('off')
                plt.imshow(viz_feature)
           
                fig.add_subplot(gs01[0])
                viz_ortho = torch.norm(ortho_feats, dim=1)
                viz_ortho = (viz_ortho).detach().cpu().numpy()[0]
                plt.title('C%d ortho feature'%(cam+1))
                plt.axis('off')
                # plt.imshow(grid_rot180(viz_ortho))
                plt.imshow(viz_ortho)
              
                
                fig.add_subplot(gs01[1])
                viz_fuse_ortho = torch.norm(ortho, dim=1) 
                viz_fuse_ortho = (viz_fuse_ortho).detach().cpu().numpy()[0]
                # plt.imshow(grid_rot180(viz_fuse_ortho))
                plt.imshow(viz_fuse_ortho)
                plt.title('After fusing C%d ortho feature'%(cam+1))
                plt.axis('off')
                plt.show()

        # Apply topdown network to fuse features from different perspectives
        # topdown = self.topdown(ortho) Discarded, topdown layer make model hard to train
        topdown = ortho

        # Predict outputs
        fuse_feature = self.fuse(topdown)
        heatmap = self.map_classifier(fuse_feature)
        tytx = self.tytx_pred(topdown)
        if self.mode == '3D':
            orient = self.orient_pred(fuse_feature)
            thtwtl = self.thtwtl_pred(topdown)

            encoded_pred = {'heatmap' : heatmap,
                            'loc_offset' : tytx.permute(0, 2, 3, 1),
                            'dim_offset' : thtwtl.permute(0, 2, 3, 1),
                            'rotation' : orient.permute(0, 2, 3, 1)}
            return encoded_pred
        elif self.mode == '2D':
            encoded_pred = {'heatmap' : heatmap,
                            'loc_offset' : tytx.permute(0, 2, 3, 1)}
            return encoded_pred

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from moft.utils import collate, grid_rot180
    from moft.data.dataset import frameDataset, MultiviewC, Wildtrack
    from moft.data.encoder import ObjectEncoder
    from moft.utils import to_numpy
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from train import parse, mx_opts, mc_opts, wt_opts
    args = parse(wt_opts)
    import numpy as np

    train_transform = transforms.Compose([transforms.Resize(args.resize_size),
                                          transforms.ToTensor()])

    # ck_p = r'F:\ANU\ENGN8602\Code\moft3d\experiments\2021-10-22_15-06-31\checkpoints\Epoch30_train_loss0.0362_val_loss0.4033.pth'
    dataset = frameDataset(MultiviewC(root=r'F:\ANU\ENGN8602\Data\MultiviewC_github\dataset'), transform=train_transform)
    encoder = ObjectEncoder(dataset)
    
    # state_dict = torch.load(ck_p)
    # [NOTICE] MultiviewX need to adjust the size of voxel of 3D grid we design 
    # The grid height is vitally important. If gird_height is too height, unvalid feature will be extracted such as the feature of sky.
    # model = MOFTNet(args=args, grid_height=64, cube_size=[4, 4, 8], mode='2D', pretrained=False) 
    model = MOFTNet(args=args, grid_height=160, cube_size=[25, 25, 32], mode='3D', pretrained=False) 
    # model.load_state_dict(state_dict['model_state_dict'])

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate)
    index, images, objects, heatmaps, calibs, grid = next(iter(dataloader))

    encoded_gt = encoder.batch_encode(objects, heatmaps, grid)[0]
    gt_heatmap = (encoded_gt['heatmap'][0, 0].detach().cpu().numpy() * 255).astype(np.uint8)

    encoded_pred = model(images, calibs, grid, visualize=True, visualize_ortho=False)
     
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.axis('off')
    # plt.imshow(grid_rot180(gt_heatmap))
    plt.imshow(gt_heatmap)

    pred_heatmap = torch.sigmoid(encoded_pred['heatmap'])
    pred_heatmap = (pred_heatmap[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    plt.subplot(122)
    plt.axis('off')
    # plt.imshow(grid_rot180(pred_heatmap))
    plt.imshow(pred_heatmap)

    plt.show()

