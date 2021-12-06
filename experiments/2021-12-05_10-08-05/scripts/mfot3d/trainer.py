import time, torch, os
from tqdm import tqdm

from vfa.model.loss import compute_loss3d, compute_loss2d
from vfa.utils import MetricDict, record
from vfa.visualization.figure import visualize_image, visualize_heatmap, visualize_bboxes, visualize_bottom
class Trainer(object):
    def __init__(self, model, args, device, summary, loss_weight=[1., 1., 1., 1.]):
        self.model = model
        self.args = args
        self.device = device
        self.summary = summary
        self.loss_weight = loss_weight
        self.mode = args.mode

    def train(self, dataloader, encoder, optimizer, epoch, args):
        self.model.train()
        epoch_loss = MetricDict()
        t_b = time.time()
        t_forward, t_backward = 0, 0
        with tqdm(total=len(dataloader), desc=f'\033[33m[TRAIN]\033[0m Epoch {epoch} / {args.epochs}', postfix=dict, mininterval=0.2) as pbar:
            for idx, (_, images, objects, heatmaps, calibs, grid) in enumerate(dataloader):
                images, calibs, heatmaps, grid = images.to(self.device), calibs.to(self.device), heatmaps.to(self.device), grid.to(self.device)
                
               
                encoded_pred = self.model(images, calibs, grid)
                
                t_f = time.time()
                t_forward += t_f - t_b

                encoded_gt = encoder.batch_encode(objects, heatmaps, grid)[0]

                if self.mode == '3D':
                    loss, loss_dict = compute_loss3d(encoded_pred, encoded_gt, self.loss_weight)
                elif self.mode == '2D':
                    loss, loss_dict = compute_loss2d(encoded_pred, encoded_gt, self.loss_weight)
                
                epoch_loss += loss_dict

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t_b = time.time()
                t_backward += t_b - t_f

                if idx % args.print_iter == 0:
                    mean_loss = epoch_loss.mean
                    if self.mode == '3D':
                        pbar.set_postfix(**{
                            '(1)loss_total' : '\033[33m{:.6f}\033[0m'.format(mean_loss['loss']), 
                            '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                            '(3)loss_pos' : '{:.5}'.format(mean_loss['loss_pos']),
                            '(4)loss_hwl': '{:.5}'.format(mean_loss['loss_hwl']),
                            '(5)loss_ang' : '\033[33m{:.5}\033[0m'.format(mean_loss['loss_ang']),
                            '(6)t_f & t_b' : '{:.2f} & {:.2f}'.format(t_forward/(idx+1), t_backward/(idx+1))
                            }
                        )
                    elif self.mode == '2D':
                        pbar.set_postfix(**{
                            '(1)loss_total' : '\033[33m{:.6f}\033[0m'.format(mean_loss['loss']), 
                            '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                            '(3)loss_pos' : '{:.5}'.format(mean_loss['loss_pos']),
                            '(4)t_f & t_b' : '{:.2f} & {:.2f}'.format(t_forward/(idx+1), t_backward/(idx+1))
                            }
                        )
                    pbar.update(1)
                if idx % args.vis_iter == 0:
                    steps = (epoch-1) * (len(dataloader) // args.vis_iter) + idx // args.vis_iter
                    if self.mode == '3D':
                        # Decode prediction
                        preds = encoder.batch_decode(encoded_pred, args.cls_thresh)
                        self.summary.add_figure('train/bboxes',
                                    visualize_bboxes(images[0], calibs[0], objects[0], preds), steps)
                    elif self.mode == '2D':
                        # Decode prediction
                        preds = encoder.batch_decode(encoded_pred, args.cls_thresh)
                        self.summary.add_figure('train/bboxes',
                                    visualize_bottom(images[0], calibs[0], objects[0], preds, args), steps)
                                    
                    # Visualize image
                    self.summary.add_image('train/image', visualize_image(images[0]), steps)
                    # Visualize heatmap
                    self.summary.add_figure('train/heatmap', 
                                visualize_heatmap(torch.sigmoid(encoded_pred['heatmap']), encoded_gt['heatmap']), steps)
                    record(os.path.join(args.savedir, 'loss', 'train_loss.txt'),
                          'Epoch:{}, Steps:{}, loss:{:.5f}, loss_heatmap:{:.5f}\n'.format(epoch, steps, epoch_loss.mean['loss'], epoch_loss.mean['loss_heatmap']))
        return epoch_loss.mean      

    def validate(self, dataloader, encoder, epoch, args):
        self.model.eval()
        epoch_loss = MetricDict()
        t_b = time.time()
        t_forward, t_backward = 0, 0
        with tqdm(total=len(dataloader), desc=f'\033[31m[VAL]\033[0m Epoch {epoch} / {args.epochs}', postfix=dict, mininterval=3) as pbar:
            for idx, (_, images, objects, heatmaps, calibs, grid) in enumerate(dataloader):
                with torch.no_grad():
                    images, calibs, heatmaps, grid = images.to(self.device), calibs.to(self.device), heatmaps.to(self.device), grid.to(self.device)
                
                    encoded_pred = self.model(images, calibs, grid)
                    
                    t_f = time.time()
                    t_forward += t_f - t_b

                    encoded_gt = encoder.batch_encode(objects, heatmaps, grid)[0]

                    if self.mode == '3D':
                        _, loss_dict = compute_loss3d(encoded_pred, encoded_gt, self.loss_weight)
                    elif self.mode == '2D':
                        _, loss_dict = compute_loss2d(encoded_pred, encoded_gt, self.loss_weight)
                    epoch_loss += loss_dict

                    t_b = time.time()
                    t_backward += t_b - t_f

                    if idx % args.print_iter == 0:
                        mean_loss = epoch_loss.mean
                        if self.mode == '3D':
                            pbar.set_postfix(**{
                                '(1)loss_total' : '\033[31m{:.6f}\033[0m'.format(mean_loss['loss']), 
                                '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                                '(3)loss_pos' : '{:.5}'.format(mean_loss['loss_pos']),
                                '(4)loss_hwl': '{:.5}'.format(mean_loss['loss_hwl']),
                                '(5)loss_ang' : '\033[31m{:.5}\033[0m'.format(mean_loss['loss_ang']),
                                '(6)t_f & t_b' : '{:.2f} & {:.2f}'.format(t_forward/(idx+1), t_backward/(idx+1))
                                }
                            )
                        elif self.mode == '2D':
                            pbar.set_postfix(**{
                                '(1)loss_total' : '\033[31m{:.6f}\033[0m'.format(mean_loss['loss']), 
                                '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                                '(3)loss_pos' : '{:.5}'.format(mean_loss['loss_pos']),
                                '(4)t_f & t_b' : '{:.2f} & {:.2f}'.format(t_forward/(idx+1), t_backward/(idx+1))
                                }
                            )
                        pbar.update(1)

        return epoch_loss.mean 


