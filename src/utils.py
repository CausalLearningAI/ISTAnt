import torch
import random
import os

def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_time_components(total_seconds):
    days = total_seconds // (24 * 3600)
    total_seconds %= (24 * 3600)
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    total_seconds %= 60
    seconds = total_seconds
    return int(days), int(hours), int(minutes), int(seconds)

def get_time_string(total_seconds):
    days, hours, minutes, seconds = get_time_components(total_seconds)
    time_str = ''
    if days > 0:
        time_str += f'{days}d '
    if hours > 0:
        time_str += f'{hours}h'
    if minutes > 0:
        time_str += f'{minutes}m'
    time_str += f'{seconds}s'
    return time_str

def get_metric(Y, Y_hat, metric="accuracy"):
        if metric == "accuracy":
            metric =  (Y_hat == Y).float().mean()
        elif metric == "balanced_acc":
            TP = ((Y == 1) & (Y_hat == 1)).sum()
            FP = ((Y != 1) & (Y_hat == 1)).sum()
            FN = ((Y == 1) & (Y_hat != 1)).sum()
            TN = ((Y != 1) & (Y_hat != 1)).sum()
            recall = TP / (TP + FN)
            specificy = TN / (TN + FP)
            metric = (recall+specificy)/2
        elif metric == "bias":
            metric =  ((Y_hat-Y)**2).mean()
        elif metric == "overestimate":
            FP = ((Y != 1) & (Y_hat == 1)).sum()
            FN = ((Y == 1) & (Y_hat != 1)).sum()
            metric = (FP-FN)/len(Y)
        elif metric == "tr_equality":
            FP = ((Y != 1) & (Y_hat == 1)).sum()
            FN = ((Y == 1) & (Y_hat != 1)).sum()
            metric =  FN / FP
        else:
            raise ValueError(f"Metric '{metric}' not implented.")
        return metric.item()