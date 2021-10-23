import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred_heatmap, gt_heatmap, alpha=2., beta=4., eps=1e-5, reduction='mean'):
    """
        Focal loss function for heatmap 
    """
    pred = torch.sigmoid(pred_heatmap).clamp(eps, 1. - eps)
    
    positive_mask = gt_heatmap == 1.
    negative_mask = ~positive_mask

    positive_num = positive_mask.sum()
    negative_num = negative_mask.sum()

    positive_loss = - (((1 - pred) ** alpha) * torch.log(pred)) * positive_mask.float()
    negative_loss = - (((1 - gt_heatmap) ** beta) * (pred ** alpha) * torch.log(1 - pred)) * negative_mask.float()
    if reduction == 'mean':
        positive_loss = torch.sum(positive_loss) / positive_num
        negative_loss = torch.sum(negative_loss) / negative_num
    elif reduction == 'sum':
        positive_loss = torch.sum(positive_loss) 
        negative_loss = torch.sum(negative_loss) 
    
    if positive_num == 0:
        return negative_loss
    elif negative_num == 0:
        return positive_loss
    else:
        return negative_loss + positive_loss

def csl_angle_focal_loss(pred, gt, foreground, alpha=2., beta=4., eps=1e-5, reduction='mean'):
    """
        Focal loss function for CSL angle prediction
    """
    # Only focus on the positive samples' angle prediction
    mask = (foreground.squeeze(0) == 1.) # foreground size: (1, 1, L, W)
    pred = pred[mask]
    gt = gt[mask]

    return focal_loss(pred, gt, alpha, beta, eps, reduction)


def compute_loss3d(batch_pred, batch_gt, loss_weight=[1., 1., 1., 1.]):
    hm_weight, pos_weight, dim_weight, ang_weight = loss_weight

    loss_offset_yx_function = nn.SmoothL1Loss(reduction='none')
    loss_offset_hwl_function = nn.SmoothL1Loss(reduction='none')
    
    batch_loss_offset_yx = loss_offset_yx_function(torch.sigmoid(batch_pred['loc_offset']), batch_gt['loc_offset']) * batch_gt['mask'].squeeze(0).unsqueeze(-1)

    batch_loss_offset_hwl = loss_offset_hwl_function(batch_pred['dim_offset'], batch_gt['dim_offset']) * batch_gt['mask'].squeeze(0).unsqueeze(-1)

    batch_loss_heatmap = focal_loss(batch_pred['heatmap'], batch_gt['heatmap'], reduction='mean')
    batch_loss_angle = csl_angle_focal_loss(batch_pred['rotation'], batch_gt['rotation'], batch_gt['mask'], reduction='mean')

    batch_num_positive_samples = batch_gt['mask'].sum()
    batch_num_positive_samples = torch.maximum(batch_num_positive_samples, torch.ones_like(batch_num_positive_samples))

    batch_loss_offset_yx /= batch_num_positive_samples
    batch_loss_offset_hwl /= batch_num_positive_samples

    batch_loss_offset_yx = torch.sum(batch_loss_offset_yx)
    batch_loss_offset_hwl = torch.sum(batch_loss_offset_hwl)
    loss = batch_loss_offset_yx * pos_weight+ batch_loss_offset_hwl * dim_weight \
           + batch_loss_heatmap * hm_weight + batch_loss_angle * ang_weight
    loss_dict = {
            'loss' : loss.item(),
            'loss_heatmap' : batch_loss_heatmap.item() * hm_weight,
            'loss_pos' : batch_loss_offset_yx.item() * pos_weight,
            'loss_hwl' : batch_loss_offset_hwl.item() * dim_weight,
            'loss_ang' : batch_loss_angle.item() * ang_weight
    }
    return loss, loss_dict

def compute_loss2d(batch_pred, batch_gt, loss_weight=[1., 1.]):
    hm_weight, pos_weight = loss_weight

    loss_offset_yx_function = nn.SmoothL1Loss(reduction='none')  
    
    batch_loss_offset_yx = loss_offset_yx_function(torch.sigmoid(batch_pred['loc_offset']), batch_gt['loc_offset']) * batch_gt['mask'].squeeze(0).unsqueeze(-1)

    batch_loss_heatmap = focal_loss(batch_pred['heatmap'], batch_gt['heatmap'], reduction='mean')
    # batch_loss_heatmap = F.mse_loss(batch_pred['heatmap'], batch_gt['heatmap'])


    batch_num_positive_samples = batch_gt['mask'].sum()
    batch_num_positive_samples = torch.maximum(batch_num_positive_samples, torch.ones_like(batch_num_positive_samples))

    batch_loss_offset_yx /= batch_num_positive_samples


    batch_loss_offset_yx = torch.sum(batch_loss_offset_yx)

    loss = batch_loss_offset_yx * pos_weight + batch_loss_heatmap * hm_weight 
    loss_dict = {
            'loss' : loss.item(),
            'loss_heatmap' : batch_loss_heatmap.item() * hm_weight,
            'loss_pos' : batch_loss_offset_yx.item() * pos_weight,
    }
    return loss, loss_dict