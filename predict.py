import torch, os
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 

from vfa.utils import collate, grid_rot180
from vfa.model.vfanet import VFANet
from vfa.data.encoder import ObjectEncoder
from vfa.data.dataset import MultiviewC, frameDataset
from vfa.visualization.figure import visualize_bboxes

def parse():
    parser = ArgumentParser()

    #Data options
    parser.add_argument('--root', type=str, default=r'F:\ANU\ENGN8602\Data\MultiviewC_dataset',
                        help='root directory of MultiviewC dataset')

    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for training. [NOTICE]: this repo only support \
                              batch size of 1')
    #Model options
    parser.add_argument('--savedir', type=str,
                        default='experiments')
    
    parser.add_argument('--resume', type=str,
                        default='2021-10-12_09-50-09')
    
    parser.add_argument('--checkpoint', type=str,
                        default='Epoch39_train_loss0.0267_val_loss1.0070.pth')

    parser.add_argument('--resume_dir', type=str,
                        default=r'F:\ANU\ENGN8602\Code\moft3d\experiments\2021-10-12_09-50-09\checkpoints\Epoch39_train_loss0.0267_val_loss1.0070.pth')
    
    #Predict options
    parser.add_argument('--cls_thresh', type=float, default=0.9,
                        help='positive sample confidence threshold')  


    args = parser.parse_args()
    print('Settings:')
    print(vars(args))
    return args

def resume(resume_dir, model):
    checkpoints = torch.load(resume_dir)

    model.load_state_dict(checkpoints['model_state_dict'])

    print("Model resume training from %s" %resume_dir)

    return model


def main():
    # Parse argument
    args = parse()

    # Data
    dataset = frameDataset(MultiviewC(root=args.root), split='val')

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)  
    
    # Device: default 1 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    # Build model
    model = VFANet().to(device)             

    # Create encoder
    encoder = ObjectEncoder(dataset)   

    # Resume
    
    model = resume(args.resume_dir, model)

    # Predict
    _, images, objects, calibs, grid = next(iter(dataloader))
    images, calibs, grid = images.to(device), calibs.to(device), grid.to(device)
    
    # Batch gt encode & visualize heatmap
    encoded_gt = encoder.batch_encode(objects, grid)[0]
    gt_heatmap = (encoded_gt['heatmap'][0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    plt.subplot(121)
    plt.imshow(grid_rot180(gt_heatmap))
    
    # Predict
    encoded_pred = model(images, calibs, grid)
    preds = encoder.batch_decode(encoded_pred, args.cls_thresh)

    # Batch gt encode & visualize heatmap
    pred_heatmap = torch.sigmoid(encoded_pred['heatmap'])
    pred_heatmap = (pred_heatmap[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    plt.subplot(122)
    plt.imshow(grid_rot180(pred_heatmap))
    plt.show()

    # visualize bboxes
    for cam in range(dataset.num_cam):
        fig = visualize_bboxes(images[cam], calibs[cam], objects[0], preds)
        plt.show()

if __name__ == '__main__':
    main()