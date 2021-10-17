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
from moft.data.dataset import frameDataset, MultiviewC
from moft.data.encoder import ObjectEncoder

def parse():
    parser = ArgumentParser()

    #Data options
    parser.add_argument('--root', type=str, default=r'\Data\MultiviewC_dataset',
                        help='root directory of MultiviewC dataset')
    
    parser.add_argument('--world_size', type=int, nargs=2, default=(3900, 3900), 
                        help='width and length of designed grid')

    parser.add_argument('--cube_LW', type=int, default=25, 
                        help='width and length of each cube of designed grid')

    parser.add_argument('--cube_H', type=int, default=32, 
                        help='height of each cube of designed grid')
    
    parser.add_argument('--image_size', type=int, nargs=2, default=(720, 1080),
                        help='height and width of image')
    
    parser.add_argument('--ann', type=str, default='annotations',
                        help='annotation of MultiviewC dataset')
                        
    parser.add_argument('--calib', type=str, default='calibrations',
                        help='calibrations of MultiviewC dataset')

    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=40,
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
    parser.add_argument('--grid_h', type=int, default=160, 
                    help='height of designed grid')
    
    parser.add_argument('--cube_size', type=int, default=(25, 25, 32), 
                        help='the size of cube of designed grid')

    parser.add_argument('--topdown', type=int, default=0,
                        help='the number of residual blocks in topdown network')

    parser.add_argument('--angle_range', type=int, default=360,
                        help='the range of angle prediction for circle smooth label (CSL)')

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
    parser.add_argument('--loss_weight', type=float, nargs=4, default=[2, 1., 1., 1],
                        help='the weight of each loss including heatmap, location, dimension and rotation')

    parser.add_argument('--print_iter', type=int, default=1,
                        help='print loss summary every N iterations')

    parser.add_argument('--vis_iter', type=int, default=50,
                        help='display visualizations every N iterations')

    parser.add_argument('--cls_thresh', type=float, default=0.5,
                        help='positive sample confidence threshold')                        

    parser.add_argument('--topk', type=int, default=50,
                        help='the number of positive samples after nms')                        
    
    parser.add_argument('--start_save', type=int, default=10,
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

def train():
    # Parse commond argument
    args = parse()
    
    # Setup random seed
    setup_seed(args.seed)

    #TODO: Add view-coherent data augmentation
    # Data augmentaion for training dataset 
    train_transform = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                                    transforms.ToTensor()])
    # Create datasets
    train_data = frameDataset(MultiviewC(root=args.root, ann_root=args.ann, calib_root=args.calib), transform=train_transform,
                              world_size=args.world_size, cube_LW=args.cube_LW, cube_H=args.cube_H, split='train')
    
    val_data = frameDataset(MultiviewC(root=args.root, ann_root=args.ann, calib_root=args.calib),\
                              world_size=args.world_size, cube_LW=args.cube_LW, cube_H=args.cube_H, split='val')

    # Create dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Device: default 1 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = MOFTNet(grid_height=args.grid_h, cube_size=args.cube_size, 
                    angle_range=args.angle_range).to(device)

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
        if epoch > args.start_save:
            save(model, epoch, args, optimizer, scheduler, train_loss, val_loss)


if __name__ == '__main__':
    train()
        
