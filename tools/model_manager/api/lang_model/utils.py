import os
import torch
from pathlib import Path

def read_syscalls(filepath):
    with open(filepath, 'r') as f:
        syscalls = f.read().split()
        return syscalls

def read_corpus(corpus_root, max_size=None):
    res = []
    files = sorted(Path(corpus_root).iterdir(), key=os.path.getmtime, reverse=True)
    for i, filename in enumerate(files):
        if max_size and i > max_size: break
        filepath = os.path.join(corpus_root, filename)
        sent = []
        with open(filepath, 'r') as f:
            for line in f.read().splitlines():
                tokens = line.split(" = ")
                if len(tokens) !=2 and len(tokens) != 1:
                    print(f'Exception: {line}, {tokens}')
                idx = tokens[-1].index('(')
                target = tokens[-1][:idx]
                sent.append(target)
        if len(sent) > 1:
            res.append(sent)
    return res

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, label, lengths, topk, device):
    maxk = max(topk) 
    total = torch.sum(lengths).item()

    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t() 
    
    pad_mask = torch.zeros(label.shape).to(device)
    pad_mask = ~pad_mask.eq(label)
    
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    correct = torch.logical_and(correct, pad_mask.view(1, -1).expand_as(pred)).contiguous()
    rtn = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) 
        rtn.append(correct_k.mul_(100.0 / total)) 
    return rtn

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
