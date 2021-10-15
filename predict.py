import torch, os
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 

from moft.utils import collate, grid_rot180
from moft.model.oftnet import MOFTNet
from moft.data.encoder import ObjectEncoder
from moft.data.dataset import MultiviewC, frameDataset
from moft.visualization.figure import visualize_bboxes

def parse():
    parser = ArgumentParser()

    #Data options
    parser.add_argument('--root', type=str, default=r'\Data\MultiviewC_dataset',
                        help='root directory of MultiviewC dataset')

    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for training. [NOTICE]: this repo only support \
                              batch size of 1')
    #Model options
    parser.add_argument('--savedir', type=str,
                        default='experiments')
    
    parser.add_argument('--resume', type=str,
                        default='2021-10-14_20-15-45')
    
    parser.add_argument('--checkpoint', type=str,
                        default='Epoch32_train_loss0.0075_val_loss0.7319.pth')
    
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
    model = MOFTNet().to(device)             

    # Create encoder
    encoder = ObjectEncoder(dataset)   

    # Resume
    resume_dir = os.path.join(args.savedir, args.resume, 'checkpoints', args.checkpoint)      
    model = resume(resume_dir, model)

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