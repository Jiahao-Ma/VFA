from genericpath import exists
import os, random
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torchvision import transforms
from distutils.dir_util import copy_tree

from moft.model.oftnet import MOFTNet
from moft.trainer import Trainer
from torch.utils.data import DataLoader
from moft.utils import collate
from moft.data.dataset import frameDataset, MultiviewC, MultiviewX
from moft.data.encoder import ObjectEncoder
from moft.config import *

def parse(opts):

    parser = ArgumentParser()

    #Data options
    parser.add_argument('--root', type=str, default=opts.root,
                        help='root directory of MultiviewC dataset')

    parser.add_argument('--mode', type=str, default=opts.mode,
                        help='2D/3D mode determines the detection task.')
    
    # MultiviewC: (3900, 3900), MultiviewX: (640, 1000)
    parser.add_argument('--world_size', type=int, nargs=2, default=opts.world_size, 
                        help='width and length of designed grid')

    # MultiviewC: (720, 1080), MultiviewX: (1080, 1920)
    parser.add_argument('--image_size', type=int, nargs=2, default=opts.image_size,
                        help='height and width of image')
    
    parser.add_argument('--resize_size', type=int, nargs=2, default=opts.resize_size,
                        help='resized height and width of image')
    
    
    parser.add_argument('--ann', type=str, default=opts.ann,
                        help='annotation of MultiviewC dataset')
                        
    parser.add_argument('--calib', type=str, default=opts.calib,
                        help='calibrations of MultiviewC dataset')

    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='the number of epochs for training')
    
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for training. [NOTICE]: this repo only support \
                              batch size of 1')

    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='learning rate')
    
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    

    # Model options
    # MultiviewC: 160, MultiviewX: 32
    parser.add_argument('--grid_h', type=int, default=opts.grid_h, 
                    help='height of designed grid')

    # MultiviewC: (25, 25, 32), MultiviewX: (4, 4, 4)
    parser.add_argument('--cube_size', type=int, default=opts.cube_size,  
                        help='the size of cube of designed grid')

    parser.add_argument('--grid_scale', type=int, default=opts.grid_scale,
                        help='make the ratio and scale of grid correspond, \
                              which also project the design voxel to image successfully.')

    parser.add_argument('--topdown', type=int, default=0, # discarded
                        help='the number of residual blocks in topdown network')

    parser.add_argument('--angle_range', type=int, default=360,
                        help='the range of angle prediction for circle smooth label (CSL)')

    parser.add_argument('--pretrained', type=bool, default=True,
                        help='load the pretrained checkpoint of feature extractor eg. resnet18')                        

    # Training options
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed')

    parser.add_argument('--savedir', type=str,
                        default='experiments')
    
    parser.add_argument('--resume', type=str,
                        default=None)
    
    parser.add_argument('--checkpoint', type=str,
                        default=None)

    # Experiment options
    # MultiviewC 3D detection: heatmap, location, dimension and rotation. loss_weight has 4 weights in total.
    # MultiviewX 2D detection: heatmap and location. loss_weight has 2 weights in total.
    parser.add_argument('--loss_weight', type=float, nargs=4, default=opts.loss_weight,
                        help='the 3D weight of each loss including heatmap, location, dimension and rotation;\
                             or 2D weight of each loss only including heatmap and location.')

    parser.add_argument('--print_iter', type=int, default=1,
                        help='print loss summary every N iterations')

    parser.add_argument('--vis_iter', type=int, default=50,
                        help='display visualizations every N iterations')

    parser.add_argument('--cls_thresh', type=float, default=0.8,
                        help='positive sample confidence threshold')                        

    parser.add_argument('--topk', type=int, default=50,
                        help='the number of positive samples after nms')                        
    
    parser.add_argument('--start_save', type=int, default=5,
                        help='After `start_save` epochs, model starts to save.')
    
    parser.add_argument('--copy_repo', type=bool, default=True,
                        help='Copy the whole repo before training')


    args = parser.parse_args()
    print('Settings:')
    print(vars(args))
    return args

def setup_seed(seed=7777):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def make_experiment(args, copy_repo=False):
    lastdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # lastdir = 'TestRepo'
    args.savedir = os.path.join(args.savedir , lastdir)
    summary = SummaryWriter(args.savedir+'/tensorboard')
    summary.add_text('config', '\n'.join(
        '{:12s} {}'.format(k, v) for k, v in sorted(args.__dict__.items())))
    summary.file_writer.flush()
    if copy_repo:
        os.makedirs(args.savedir, exist_ok=True)
        copy_tree('./moft', args.savedir + '/scripts/mfot3d')
    return summary, args

def resume_experiment(args):
    summary_dir = os.path.join(args.savedir, args.resume, 'tensorboard')
    args.savedir = os.path.join(args.savedir, args.resume)
    summary = SummaryWriter(summary_dir)
    return summary, args

def save(model, epoch, args, optimizer, scheduler, train_loss, val_loss):
    savedir = os.path.join(args.savedir, 'checkpoints')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    checkpoints = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict()
    }
    torch.save(checkpoints, os.path.join(savedir, 'Epoch{:02d}_train_loss{:.4f}_val_loss{:.4f}.pth'.\
                        format(epoch, train_loss['loss'], val_loss['loss'])))

def resume(resume_dir, model, optimizer, scheduler, device):
    checkpoints = torch.load(resume_dir)
    pretrain = checkpoints['model_state_dict']
    current = model.state_dict()
    state_dict = {k: v for k, v in pretrain.items() if k in current.keys()}
    current.update(state_dict)
    model.load_state_dict(current)

    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    epoch = checkpoints['epoch'] + 1
    print("Model resume training from %s" %resume_dir)
    return model, optimizer, scheduler, epoch

def train(opts):
    # Parse commond argument
    args = parse(opts)
    
    # Setup random seed
    setup_seed(args.seed)

    #TODO: Add view-coherent data augmentation
    # Data augmentaion for training dataset 
    train_transform = transforms.Compose([transforms.Resize(args.resize_size),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                                          transforms.ToTensor()])
    # Create datasets
    if opts.name == 'MultiviewC':
        train_data = frameDataset(MultiviewC(root=args.root, ann_root=args.ann, calib_root=args.calib, 
                                             world_size=args.world_size, cube_LWH=args.cube_size), 
                                             transform=train_transform, split='train')
        
        val_data = frameDataset(MultiviewC(root=args.root, ann_root=args.ann, calib_root=args.calib, 
                                           world_size=args.world_size, cube_LWH=args.cube_size),
                                           split='val')
    elif opts.name == 'MultiviewX':
        train_data = frameDataset(MultiviewX(root=args.root, world_size=args.world_size, cube_LWH=args.cube_size), 
                                             transform=train_transform, split='train')
        
        val_data = frameDataset(MultiviewX(root=args.root, world_size=args.world_size, cube_LWH=args.cube_size),
                                           split='val')

    # Create dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Device: default 1 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    if args.model_type == 'moftnet':
        model = MOFTNet(grid_height=args.grid_h, cube_size=args.cube_size, angle_range=args.angle_range,
                        grid_scale=args.grid_scale, mode=args.mode, pretrained=args.pretrained).to(device)

    # Create encoder
    encoder = ObjectEncoder(train_data, topk=args.topk)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), 
                                              epochs=args.epochs)

    # Create Summary & Resume Training
    if args.resume is not None:
        summary, args = resume_experiment(args)
        resume_dir = os.path.join(args.savedir, 'checkpoints', args.checkpoint)
        model, optimizer, scheduler, start = \
            resume(resume_dir, model, optimizer, scheduler, device)
    else:
        summary, args = make_experiment(args, args.copy_repo)
        start = 1

    # Create Trainer
    trainer = Trainer(model, args, device, summary, args.loss_weight)

    for epoch in range(start, args.epochs+1):
        scheduler.step()
        summary.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Train model
        train_loss = trainer.train(train_loader, encoder, optimizer, epoch, args)    
        
        # Validate model
        val_loss = trainer.validate(val_loader, encoder, epoch, args)

        summary.add_scalars('loss', {'train_loss': train_loss['loss'], 'val_loss' : val_loss['loss']}, epoch)
        # if epoch > args.start_save:
        if epoch % 5 == 0:
            save(model, epoch, args, optimizer, scheduler, train_loss, val_loss)


if __name__ == '__main__':

    # MultiviewC
    # train(mc_opts)

    # MultiviewX 
    train(mx_opts)

    # os.system('shutdown /s /t 10')
        
