import os
import time
import shutil

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter
import sklearn
import sod_metric
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # if flatten:
    #     ret = ret.view(-1, ret.shape[-1])

    return ret



def calc_cod(y_pred, y_true):
    batchsize = y_true.shape[0]

    metric_FM = sod_metric.Fmeasure()
    metric_WFM = sod_metric.WeightedFmeasure()
    metric_SM = sod_metric.Smeasure()
    metric_EM = sod_metric.Emeasure()
    metric_MAE = sod_metric.MAE()
    with torch.no_grad():
        assert y_pred.shape == y_true.shape

        for i in range(batchsize):
            true, pred = \
                y_true[i, 0].cpu().data.numpy() * 255, y_pred[i, 0].cpu().data.numpy() * 255

            metric_FM.step(pred=pred, gt=true)
            metric_WFM.step(pred=pred, gt=true)
            metric_SM.step(pred=pred, gt=true)
            metric_EM.step(pred=pred, gt=true)
            metric_MAE.step(pred=pred, gt=true)

        fm = metric_FM.get_results()["fm"]
        wfm = metric_WFM.get_results()["wfm"]
        sm = metric_SM.get_results()["sm"]
        em = metric_EM.get_results()["em"]["curve"].mean()
        mae = metric_MAE.get_results()["mae"]

    return sm, em, wfm, mae


from sklearn.metrics import precision_recall_curve


# def calc_f1(y_pred,y_true):
#     batchsize = y_true.shape[0]
#     with torch.no_grad():
#         assert y_pred.shape == y_true.shape
#         f1, auc = 0, 0
#         y_true = y_true.cpu().numpy()
#         y_pred = y_pred.cpu().numpy()
#         for i in range(batchsize):
#             true = y_true[i].flatten()
#             true = true.astype(np.int)
#             pred = y_pred[i].flatten()

#             precision, recall, thresholds = precision_recall_curve(true, pred)

#             # auc
#             auc += roc_auc_score(true, pred)
#             # auc += roc_auc_score(np.array(true>0).astype(np.int), pred)
#             f1 += max([(2 * p * r) / (p + r+1e-10) for p, r in zip(precision, recall)])

#     return f1/batchsize, auc/batchsize, np.array(0), np.array(0)

def calc_f1(y_pred,y_true):
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        f1, pre = 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            true = true.astype(np.int)
            pred = y_pred[i].flatten()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = sklearn.metrics.f1_score(true, pred, pos_label=1, average='binary')

            # auc
            # auc += roc_auc_score(true, pred)
            # auc += roc_auc_score(np.array(true>0).astype(np.int), pred)
            # pre += precision
            # import ipdb; ipdb.set_trace()
            # f1 += max([(2 * p * r) / (p + r+1e-10) for p, r in zip(precision, recall)])
    
    return f1/batchsize, np.array(0), np.array(0), np.array(0)



def calc_fmeasure(y_pred,y_true):
    batchsize = y_true.shape[0]

    mae, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(batchsize):
            gt_float, pred_float = \
                y_true[i, 0].cpu().data.numpy(), y_pred[i, 0].cpu().data.numpy()

            # # MAE
            mae.append(np.sum(cv2.absdiff(gt_float.astype(float), pred_float.astype(float))) / (
                        pred_float.shape[1] * pred_float.shape[0]))
            # mae.append(np.mean(np.abs(pred_float - gt_float)))
            #
            pred = np.uint8(pred_float * 255)
            gt = np.uint8(gt_float * 255)

            pred_float_ = np.where(pred > min(1.5 * np.mean(pred), 255), np.ones_like(pred_float),
                                   np.zeros_like(pred_float))
            gt_float_ = np.where(gt > min(1.5 * np.mean(gt), 255), np.ones_like(pred_float),
                                 np.zeros_like(pred_float))

            preds.extend(pred_float_.ravel())
            gts.extend(gt_float_.ravel())

        RECALL = recall_score(gts, preds)
        PERC = precision_score(gts, preds)

        fmeasure = (1 + 0.3) * PERC * RECALL / (0.3 * PERC + RECALL)
        MAE = np.mean(mae)

    return fmeasure, MAE, np.array(0), np.array(0)

from sklearn.metrics import roc_auc_score,recall_score,precision_score
import cv2
def calc_ber(y_pred, y_true):
    batchsize = y_true.shape[0]
    y_pred, y_true = y_pred.permute(0, 2, 3, 1).squeeze(-1), y_true.permute(0, 2, 3, 1).squeeze(-1)
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        pos_err, neg_err, ber = 0, 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            pred = y_pred[i].flatten()

            TP, TN, FP, FN, BER, ACC = get_binary_classification_metrics(pred * 255,
                                                                         true * 255, 125)
            pos_err += (1 - TP / (TP + FN)) * 100
            neg_err += (1 - TN / (TN + FP)) * 100

    return pos_err / batchsize, neg_err / batchsize, (pos_err + neg_err) / 2 / batchsize, np.array(0)

def get_binary_classification_metrics(pred, gt, threshold=None):
    if threshold is not None:
        gt = (gt > threshold)
        pred = (pred > threshold)
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    return TP, TN, FP, FN, BER, ACC

def cal_ber(tn, tp, fn, fp):
    return  0.5*(fp/(tn+fp) + fn/(fn+tp))

def cal_acc(tn, tp, fn, fp):
    return (tp + tn) / (tp + tn + fp + fn)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() +
                                                                1e-20)
    return prec, recall

def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q

def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score

def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1) * round(cols / 2)
        Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).float().cuda()
        j = torch.from_numpy(np.arange(0, rows)).float().cuda()
        X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
    return X.long(), Y.long()


def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

def _eval_e(y_pred, y, num):
    score = torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score


from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from PIL import Image

def cal_iou(y_pred, y_true):
    with torch.no_grad():
        # assert y_pred.shape == y_true.shape
        res = process(y_pred, y_true)
        # import ipdb; ipdb.set_trace()
        metrics = compute_metrics(res)
    # import ipdb; ipdb.set_trace()
    # return metrics['mIoU'], metrics['mAcc'], 0, 0
    return metrics['aAcc'], metrics['mFscore'], metrics['mPrecision'], metrics['mRecall']
        
def process(pred_labels, gt_labels) -> None:
    """Process one batch of data and data_samples.

    The processed results should be stored in ``self.results``, which will
    be used to compute the metrics when all batches have been processed.

    Args:
        data_batch (dict): A batch of data from the dataloader.
        data_samples (Sequence[dict]): A batch of outputs from the model.
    """
    results = []
    num_classes = 2
    i = 0
    for pred_label, label in zip(pred_labels, gt_labels):
        b, c, h, w = pred_label.shape
        pred_label = pred_label.view(h, w)
        label = label.view(h, w)
        pred = pred_label.squeeze()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        lab = label.squeeze().to(pred_label)
        # print(pred.sum(), lab.sum())
        results.append(intersect_and_union(pred, lab, num_classes))
        # png_filename = 'img_save/img' + str(i) + '.jpg'
        # png_filename_gt = 'img_save/gt' + str(i) + '.jpg'
        # format_result
        # output_mask = pred_label.cpu().numpy()
        # output_gt_mask = label.cpu().numpy()
        
        # output = Image.fromarray(output_mask.astype(np.uint8))
        # output_gt = Image.fromarray(output_gt_mask.astype(np.uint8))
        
        # output.save(png_filename)
        # output_gt.save(png_filename_gt)
        i += 1

    return results


def compute_metrics(results: list) -> Dict[str, float]:
    """Compute the metrics from processed results.

    Args:
        results (list): The processed results of each batch.

    Returns:
        Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results. The key
            mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
            mRecall.
    """
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    results = tuple(zip(*results))
    assert len(results) == 4
    metric = ['mFscore']
    # metric = ['mIoU']
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    ret_metrics = total_area_to_metrics(
        total_area_intersect, total_area_union, total_area_pred_label,
        total_area_label, metric)

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    metrics = dict()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            metrics[key] = val
        else:
            metrics['m' + key] = val
    ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
    import ipdb; ipdb.set_trace()
    return metrics


def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                        num_classes: int):
    """Calculate Intersection and Union.

    Args:
        pred_label (torch.tensor): Prediction segmentation map
            or predict result filename. The shape is (H, W).
        label (torch.tensor): Ground truth segmentation map
            or label filename. The shape is (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
        torch.Tensor: The prediction histogram on all classes.
        torch.Tensor: The ground truth histogram on all classes.
    """

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_area_to_metrics(total_area_intersect: np.ndarray,
                          total_area_union: np.ndarray,
                          total_area_pred_label: np.ndarray,
                          total_area_label: np.ndarray,
                          metrics: List[str] = ['mIoU'],
                          nan_to_num: Optional[int] = None,
                          beta: int = 1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (np.ndarray): The intersection of prediction
            and ground truth histogram on all classes.
        total_area_union (np.ndarray): The union of prediction and ground
            truth histogram on all classes.
        total_area_pred_label (np.ndarray): The prediction histogram on
            all classes.
        total_area_label (np.ndarray): The ground truth histogram on
            all classes.
        metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
            'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be
            replaced by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
    Returns:
        Dict[str, np.ndarray]: per category evaluation metrics,
            shape (num_classes, ).
    """

    def f_score(precision, recall, beta=1):
        """calculate the f-score value.

        Args:
            precision (float | torch.Tensor): The precision value.
            recall (float | torch.Tensor): The recall value.
            beta (int): Determines the weight of recall in the combined
                score. Default: 1.

        Returns:
            [torch.tensor]: The f-score value.
        """
        score = (1 + beta ** 2) * (precision * recall) / (
                (beta ** 2 * precision) + recall)
        return score

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError(f'metrics {metrics} is not supported')

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor([
                f_score(x[0], x[1], beta) for x in zip(precision, recall)
            ])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics