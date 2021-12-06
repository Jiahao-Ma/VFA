import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import shapely
from shapely.geometry import Polygon, MultiPoint    # 多边形计算的库
from .IoU import IoU3D
import torch

def CLEAR_MOD_HUN2(gt, det, thresh):
    frames = int(max(gt[:, 0])) + 1
    all_infolist = None
    for t in range(1, frames + 1):
        gt_results = gt[gt[:, 0] == t - 1]
        det_results = det[det[:, 0] == t - 1]
        frame_infolist = cal_frame_TPFP_iou(thresh, gt_results, det_results)
        if all_infolist is None:
            all_infolist = frame_infolist
        else:
            all_infolist = np.concatenate((all_infolist, frame_infolist), axis=0)

    idx = np.argsort(all_infolist[:,0], axis=0)
    idx = idx[::-1]
    all_infolist = all_infolist[idx]
    TP = 0
    FP = 0
    all_P = gt.shape[0]
    for i, data in enumerate(all_infolist):
        flag = data[-4]
        if flag == 1:
            TP += 1
        else:
            FP += 1

        all_infolist[i, -3] = TP / (TP + FP)
        all_infolist[i, -2] = TP / all_P
        cur_aos = 0
        for m in range(i + 1):
            cur_aos += all_infolist[m, -4] * (1 + np.cos(all_infolist[m, 3])) / 2
        cur_aos /= (i + 1)
        all_infolist[i, -1] = cur_aos
    recall_threshold = np.arange(0, 1.1, 0.1)
    accu_precisions = 0
    for thresh in recall_threshold:
        max_prec = 0
        for k in range(all_infolist.shape[0]):
            if all_infolist[k][-2] >= thresh:
                max_prec = max(all_infolist[k:,-3])
                break
        accu_precisions += max_prec


    final_11_precision = accu_precisions / 11

    # AOS
    accu_aos = 0
    for thresh in recall_threshold:
        max_aos = 0
        for k in range(all_infolist.shape[0]):
            if all_infolist[k][-2] >= thresh:
                max_aos = max(all_infolist[k:,-1])
                break
        accu_aos += max_aos
    final_11_aos = accu_aos / 11

    return final_11_precision, final_11_aos


def cal_frame_TPFP_iou(dist_threshold, gt_res, pred_res):
    
    # 0     1           2           3   4       5     6       7
    # score, frame_idx, iou, delta_ori, TP/FP?, prec, recall, aos
    frame_gt_det_match = np.zeros(shape=(pred_res.shape[0], 8)) - 1
    frame_gt_det_match[:, -4:] += 1
    for i, pred in enumerate(pred_res):
        max_iou = -1
        max_idx = -1
        cur_gt_ori = -1
        _, _, pred_x, pred_y, pred_z, pred_l, pred_w, pred_h, pred_rot, conf = pred
        pred_tensor = torch.Tensor([pred_x, pred_y, pred_z, pred_l, pred_w, pred_h, pred_rot]).unsqueeze(0).unsqueeze(0).to(device=torch.device('cuda'))
        for j, gt in enumerate(gt_res):
            _, _, gt_x, gt_y, gt_z, gt_l, gt_w, gt_h, gt_rot = gt
            gt_tensor = torch.Tensor([gt_x, gt_y, gt_z, gt_l, gt_w, gt_h, gt_rot]).unsqueeze(0).unsqueeze(0).to(device=torch.device('cuda')) # default cuda
            iou = IoU3D(pred_tensor, gt_tensor)
            if max_iou != 0 and iou >= dist_threshold and iou > max_iou:
                max_iou = iou
                max_idx = j
                cur_gt_ori = gt_rot

        frame_gt_det_match[i][0] = conf
        frame_gt_det_match[i][1] = max_idx
        frame_gt_det_match[i][2] = max_iou
        frame_gt_det_match[i][3] = pred_rot - cur_gt_ori

    TP = 0
    FP = 0
    passed_index = []
    for k in range(pred_res.shape[0]):
        if -1 not in frame_gt_det_match[k, :]:
            TP += 1
            passed_index.append(frame_gt_det_match[k, :][1])
            frame_gt_det_match[k, 4] = 1
        elif -1 in frame_gt_det_match[k, :]:
            FP += 1
            frame_gt_det_match[k, 4] = 0
    return frame_gt_det_match

def evaluateDetectionAPAOS(res_fpath, gt_fpath):
    gtRaw = np.loadtxt(gt_fpath)
    detRaw = np.loadtxt(res_fpath)

    frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
    frame_ctr = 0
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0

    assert detRaw is not None and detRaw.shape[0] != 0, 'detection is empty'
        
    # pred data format: frame_id, location(x, y, z), dimension(l, w, h), rotation, conf
    # gt data format: frame_id, location(x, y, z), dimension(l, w, h), rotation
    for t in frames:
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 9))

        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)]) # frame_id
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])         # obj_id
        tmp_arr[:, 2] = np.array([x for x in gtRaw[idx, 1]])          # x
        tmp_arr[:, 3] = np.array([y for y in gtRaw[idx, 2]])          # y
        tmp_arr[:, 4] = np.array([z for z in gtRaw[idx, 3]])          # z
        tmp_arr[:, 5] = np.array([l for l in gtRaw[idx, 4]])          # l
        tmp_arr[:, 6] = np.array([w for w in gtRaw[idx, 5]])          # w
        tmp_arr[:, 7] = np.array([h for h in gtRaw[idx, 6]])          # h
        tmp_arr[:, 8] = np.array([r for r in gtRaw[idx, 7]])          # rotation

        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr), axis=0)

        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 10))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])   # frame_id
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])           # obj_id
        tmp_arr[:, 2] = np.array([x for x in detRaw[idx, 1]])           # x
        tmp_arr[:, 3] = np.array([y for y in detRaw[idx, 2]])           # y
        tmp_arr[:, 4] = np.array([z for z in detRaw[idx, 3]])           # z
        tmp_arr[:, 5] = np.array([l for l in detRaw[idx, 4]])           # l
        tmp_arr[:, 6] = np.array([w for w in detRaw[idx, 5]])           # w
        tmp_arr[:, 7] = np.array([h for h in detRaw[idx, 6]])           # h
        tmp_arr[:, 8] = np.array([r for r in detRaw[idx, 7]])           # rotation 
        tmp_arr[:, 9] = np.array([cf for cf in detRaw[idx, 8]])         # conf 
        
        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr), axis=0)
        frame_ctr += 1

    AP_75, AOS_75 = CLEAR_MOD_HUN2(gtAllMatrix, detAllMatrix, 0.75)
    AP_50, AOS_50 = CLEAR_MOD_HUN2(gtAllMatrix, detAllMatrix, 0.5)
    AP_25, AOS_25 = CLEAR_MOD_HUN2(gtAllMatrix, detAllMatrix, 0.25)
    return AP_75 * 100, AOS_75 * 100, AOS_75/AP_75, AP_50 * 100, AOS_50 * 100, AOS_50/AP_50, AP_25* 100, AOS_25* 100, AOS_25/AP_25

