import torch, os
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from tqdm import tqdm

from moft.utils import collate, to_numpy
from moft.model.oftnet import MOFTNet
from moft.data.encoder import ObjectEncoder
from moft.data.dataset import frameDataset
from moft.data.multiviewX import MultiviewX
from moft.data.multiviewC import MultiviewC
from moft.data.wildtrack import Wildtrack
from moft.visualization.figure import visualize_bboxes
from moft.evaluation.pyeval.CLEAR_MOD_HUN import CLEAR_MOD_HUN
from moft.evaluation.evaluate import evaluate_rcll_prec_moda_modp, evaluate_ap_aos
from moft.config import MultiviewX_Config, Wildtrack_Config, mx_opts, wt_opts, mc_opts
def parse(opts):
    parser = ArgumentParser()

    #Data options
    parser.add_argument('--root', type=str, default=opts.root,
                        help='root directory of dataset')

    parser.add_argument('--data', type=str, default=opts.name,
                        help='the name of dataset')                        

    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for training. [NOTICE]: this repo only support \
                              batch size of 1')
    #Model options
    parser.add_argument('--savedir', type=str,
                        default='experiments')
    
    parser.add_argument('--resume', type=str,
                        default='2021-10-28_16-31-40_wt')
    
    parser.add_argument('--checkpoint', type=str,
                        default='Epoch30_train_loss0.0176_val_loss1.4921.pth')
    
    #Predict options
    parser.add_argument('--cls_thresh', type=float, default=0.8,
                        help='positive sample confidence threshold')  

    parser.add_argument('--pr_dir_pred', type=str, 
                        default=r'F:\ANU\ENGN8602\Code\moft3d\experiments\2021-10-28_16-31-40_wt\evaluation\pr_dir_pred.txt')
    parser.add_argument('--pr_dir_gt', type=str, 
                        default=r'F:\ANU\ENGN8602\Code\moft3d\experiments\2021-10-28_16-31-40_wt\evaluation\pr_dir_gt.txt')

    parser.add_argument('--ap_aos_dir_pred', type=str, 
                        default=r'F:\ANU\ENGN8602\Code\moft3d\experiments\2021-10-28_16-31-40_wt\evaluation\ap_aos_pred.txt')
    parser.add_argument('--ap_aos_dir_gt', type=str, 
                        default=r'F:\ANU\ENGN8602\Code\moft3d\experiments\2021-10-28_16-31-40_wt\evaluation\ap_aos_gt.txt')

    parser.add_argument('--eval_mode', type=str, default='2D')

    parser.add_argument('--eval_tool', type=str, default='matlab')                        

    parser.add_argument('--config', type=Wildtrack_Config, default=opts)

    args = parser.parse_args()
    print('Settings:')
    print(vars(args))
    return args

def resume(resume_dir, device):

    checkpoints = torch.load(resume_dir)
    ck_args = checkpoints['args']
    # Build model
    model = MOFTNet(args=ck_args,
                    grid_height=ck_args.grid_h, 
                    cube_size=ck_args.cube_size,
                    mode=ck_args.mode).to(device)

    model.load_state_dict(checkpoints['model_state_dict'])

    print("Model resume training from %s" %resume_dir)

    return model


def construct_location(objects):
    locaitons = list()
    for i, obj in enumerate(objects):
        tmp = np.zeros(shape=(1, 3))
        tmp[:, 0] = i
        tmp[:, :2] = to_numpy(obj.location)[:2]
        locaitons.append(tmp)
    return np.concatenate(locaitons, axis=0)

class FormatAPAOSData():
    def __init__(self, save_dir, mode='pred') -> None:
        assert mode in ['pred', 'gt'], 'mode error'
        self.mode = mode
        self.save_dir = save_dir
        self.data = None

    def add_item(self, batch, id):
        id = np.array(id).reshape(-1)
        # construct stored data with format: frame_id, x, y, z, l, w, h, rotation, conf
        for obj in batch:
            dimension = to_numpy(obj.dimension)[::-1]
            location = to_numpy(obj.location)
            rotation = to_numpy(obj.rotation).reshape(-1)
            if self.mode == 'pred':
                conf = to_numpy(obj.conf).reshape(-1)
            if self.data is None:
                if self.mode == 'pred':
                    self.data = np.concatenate([id, location, dimension, rotation, conf], axis=0).reshape(1, -1)
                else:
                    self.data = np.concatenate([id, location, dimension, rotation], axis=0).reshape(1, -1)
            else:
                if self.mode == 'pred':
                    tmp = np.concatenate([id, location, dimension, rotation, conf], axis=0).reshape(1, -1)
                    self.data = np.vstack([self.data, tmp])
                else:
                    tmp = np.concatenate([id, location, dimension, rotation], axis=0).reshape(1, -1)
                    self.data = np.vstack([self.data, tmp])
    def save(self):
        if not os.path.exists(os.path.dirname(self.save_dir)):
            os.mkdir(os.path.dirname(self.save_dir))
        np.savetxt(self.save_dir, self.data)
    
    def exist(self):
        return os.path.exists(self.save_dir)

class FormatPRData():
    def __init__(self, save_dir) -> None:
        self.data = None
        self.save_dir = save_dir

    def add_item(self, batch, id):
        location = construct_location(batch)
        if self.data is None:
            self.data = np.concatenate([ np.ones((location.shape[0], 1))*id,  location], axis=1)
        else:
            tmp = np.concatenate([ np.ones((location.shape[0], 1))*id,  location], axis=1)
            self.data = np.concatenate([self.data, tmp], axis=0)
    def save(self):
        if not os.path.exists(os.path.dirname(self.save_dir)):
            os.mkdir(os.path.dirname(self.save_dir))
        np.savetxt(self.save_dir, self.data)
    
    def exist(self):
        return os.path.exists(self.save_dir)


def main(opts):
    # Parse argument
    args = parse(opts)

    # Data
    dataset = frameDataset(Wildtrack(root=args.root), split='val')

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)  
    
    # Device: default 1 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    # Create encoder
    encoder = ObjectEncoder(dataset)   

    # Resume
    resume_dir = os.path.join(args.savedir, args.resume, 'checkpoints', args.checkpoint)      
    model = resume(resume_dir, device)

   
    APAOS_pred = FormatAPAOSData(args.ap_aos_dir_pred, 'pred')
    APAOS_gt = FormatAPAOSData(args.ap_aos_dir_gt, 'gt')

    PR_pred = FormatPRData(args.pr_dir_pred)
    PR_gt = FormatPRData(args.pr_dir_gt)

    if not PR_pred.exist() or not PR_gt.exist() or not APAOS_pred.exist() or not APAOS_gt.exist():
        with tqdm(iterable=dataloader, desc=f'[EVALUATE] ', postfix=dict, mininterval=1) as pbar:
            for batch_idx, (_, images, objects, _, calibs, grid) in enumerate(dataloader):
                with torch.no_grad():
                    images, calibs, grid = images.to(device), calibs.to(device), grid.to(device)
                    encoded_pred = model(images, calibs, grid)
                    preds = encoder.batch_decode(encoded_pred, args.cls_thresh)

                    if args.eval_mode == '3D':
                        APAOS_pred.add_item(preds, batch_idx)
                        APAOS_gt.add_item(objects[0], batch_idx)

                    PR_pred.add_item(preds, batch_idx)
                    PR_gt.add_item(objects[0], batch_idx)
                   
                pbar.update(1)
        # Save 
        if args.eval_mode == '3D':
            APAOS_pred.save()
            APAOS_gt.save()
        PR_pred.save()
        PR_gt.save()

    recall, precision, moda, modp = evaluate_rcll_prec_moda_modp(args.pr_dir_pred, args.pr_dir_gt, dataset=args.data, eval=args.eval_tool)
    print(f'\n{args.eval_tool} eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
    if args.eval_mode == '3D':
        AP_75, AOS_75, OS_75, AP_50, AOS_50, OS_50, AP_25, AOS_25, OS_25 = evaluate_ap_aos(args.ap_aos_dir_pred, args.ap_aos_dir_gt)
        print("AP_75: %.2f" % AP_75, " ,AOS_75: %.2f" % AOS_75, ", OS_75: %.2f" % OS_75)
        print("AP_50: %.2f" % AP_50, " ,AOS_50: %.2f" % AOS_50, ", OS_50: %.2f" % OS_50)
        print("AP_25: %.2f" % AP_25, " ,AOS_25: %.2f" % AOS_25, ", OS_25: %.2f" % OS_25)

if __name__ == '__main__':
    # Wildtrack
    # main(wt_opts)

    # MultiviewX
    # main(mx_opts)
    
    # MultiviewC
    main(mc_opts)