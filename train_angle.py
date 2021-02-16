import os
import cv2
import time
import random
import pickle
import scipy
import numpy as np
from PIL import Image
from distutils.version import LooseVersion

from sacred import Experiment
from sacred.observers import FileStorageObserver
from easydict import EasyDict as edict

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf

from utils.misc import AverageMeter, get_optimizer
ex = Experiment()
ex.observers.append(FileStorageObserver.create('experiments/angle'))

from models_angle.baseline_same import Baseline as ResNet
from datasets.plane_angle_dataset import PlaneDataset

@ex.command
def train(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join('experiments/angle', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    network = ResNet(cfg.model)
    
    if not cfg.resume_dir == 'None':
        model_dict = torch.load(cfg.resume_dir)
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)

    optimizer = get_optimizer(network.parameters(), cfg.solver)

    train_dataset = PlaneDataset(cfg.dataset, split='train', random=True)
    train_loader = data.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size,
        shuffle=True, num_workers=cfg.dataset.num_workers
    )
    val_dataset = PlaneDataset(cfg.dataset, split='test', random=False, evaluation=True)
    val_loader = data.DataLoader(
        val_dataset, batch_size=36,
        shuffle=False, num_workers=cfg.dataset.num_workers
    )

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, verbose=True, min_lr=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc1 = 0.
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    for epoch in range(cfg.num_epochs):
        # train for one epoch
        network.train()
        tic = time.time()
        for iter, (image, target) in enumerate(train_loader):
            image = image.to(device)
            target = target[:,0]
            target = target.to(device)

            output = network(image)
            loss = criterion(output, target)
            losses.update(loss.item(), image.size(0))
            acc1 = accuracy(output, target)
            top1.update(acc1[0].item(), image.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - tic)
            tic = time.time()

            if iter % cfg.print_interval == 0:
                _log.info(f"[{epoch:2d}][{iter:5d}/{len(train_loader):5d}] "
                      f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                      f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                      f"Top1: {top1.val:.3f} ({top1.avg:.3f}) ")

        acc1 = validating(val_loader, network, criterion, _log, device, epoch)
        best_acc1 = max(best_acc1, acc1)
        _log.info(f"epoch: {epoch}, best accuracy: {best_acc1}")
        torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))
        #scheduler.step(acc1)


def validating(data_loader, network, criterion, log, device, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    network.eval()
    with torch.no_grad():
        for iter, (image, target) in enumerate(data_loader):
            image = image.to(device)
            target = target[:,0]
            target = target.to(device)

            output = network(image)
            loss = criterion(output, target)

            losses.update(loss.item(), image.size(0))
            acc1 = accuracy(output, target)
            top1.update(acc1[0].item(), image.size(0))
        log.info(f"Loss: {losses.avg:.4f}, Top1: {top1.avg:.4f}")

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), \
        'PyTorch 1.0.0 is used'

    ex.add_config('./configs/config_angle.yaml')
    ex.run_commandline()
