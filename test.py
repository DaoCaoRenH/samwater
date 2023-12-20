import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from PIL import Image
import torch.distributed as dist
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
from prettytable import PrettyTable

import datasets
import models
import utils



os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
os.environ["RANK"] = "0"
os.environ['WORLD_SIZE'] = '1'
# torch.distributed.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", 0)

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

def eval_psnr(loader, model, use_cuda=True):
    model.eval()
    pbar = tqdm(total=len(loader), leave=False, desc='val')

    pred_list = []
    gt_list = []
    t = []
    for batch in loader:
        if use_cuda:
            for k, v in batch.items():
                batch[k] = v.cuda()
        inp = batch['inp']
        gt = batch['gt']
        inp = model.preprocess(inp)
        start = time.time()
        with torch.no_grad():
            pred = torch.sigmoid(model.infer(inp))
            
        pred = F.interpolate(pred, (gt.shape[2], gt.shape[3]), mode="bilinear", align_corners=False)
        end = time.time()
        # t.append(np.round(end-start, 2))
        # print(end-start)
        
        batch_pred = [pred]
        batch_gt = [gt]
        
        # import ipdb; ipdb.set_trace()
        # dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        # dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    cal(pred_list, gt_list)
    # print(t)

def cal(y_pred, y_true):
    with torch.no_grad():
        res = process(y_pred, y_true)
        compute_metrics(res)

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
        results.append(intersect_and_union(pred, lab, num_classes))
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
    metric = ['mFscore', 'mIoU']
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    ret_metrics = total_area_to_metrics(
        total_area_intersect, total_area_union, total_area_pred_label,
        total_area_label, metric)
    class_names = ('background', 'water')
    ret_metrics['1-Recall'] = 1 - ret_metrics['Recall']
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
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
    
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
            
            val = [str(v) for v in val]
            class_table_data.add_column(key, val)
    table_data = PrettyTable()
    for key, val in metrics.items():
            val = [val]
            val = [str(v) for v in val]
            table_data.add_column(key, val)
    print(class_table_data.get_string())
    print(table_data.get_string())

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model']).cuda()
    # model = models.make(config['model']).cpu()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    # sam_checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(sam_checkpoint, strict=True)
    eval_psnr(loader, model, use_cuda=True)
    # eval_psnr(loader, model, use_cuda=False)
