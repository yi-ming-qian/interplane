import torch
from visdom import Visdom
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(parameters, cfg):
    if cfg.method == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                    lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters),
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters),
                                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, parameters),
                                         lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
