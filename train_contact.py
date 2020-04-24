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
from utils.metric import comp_iou
#from utils.loss import class_balanced_cross_entropy_loss
ex = Experiment()
ex.observers.append(FileStorageObserver.create('experiments/contact'))

from models_contact.baseline_same import Baseline as UNet
from datasets.plane_contact_dataset import PlaneDataset

@ex.command
def train(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join('experiments/contact', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    network = UNet(cfg.model)
    #inputs = torch.rand((3,5,224,224))
    #network(inputs)
    
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
    # for i in range(36):
    #    val_dataset[i]
    #    print(f"scene {i}")
    # exit()

    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss()
    best_acc1 = 0.
    # return
    batch_time = AverageMeter()
    losses_classify = AverageMeter()
    losses_line = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    for epoch in range(cfg.num_epochs):
        # train for one epoch
        network.train()
        tic = time.time()
        for iter, sample in enumerate(train_loader):
            image = sample["image"]
            target = sample["target"]
            gt_line = sample["gt_linemask_d"]

            image = image.to(device)
            target = target[:,0]
            target = target.to(device)
            gt_line = gt_line.to(device)

            classify_prob, line_prob= network(image)
            classify_loss = ce_loss(classify_prob, target)
            line_loss = bce_loss(line_prob, gt_line)
            loss = classify_loss + line_loss

            losses_line.update(line_loss.item())
            losses_classify.update(classify_loss.item())
            losses.update(loss.item())
            acc1 = accuracy(classify_prob, target)
            top1.update(acc1[0].item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - tic)
            tic = time.time()

            if iter % cfg.print_interval == 0:
                _log.info(f"[{epoch:2d}][{iter:5d}/{len(train_loader):5d}] "
                      f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                      f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                      f"Classify Loss: {losses_classify.val:.4f} ({losses_classify.avg:.4f}) "
                      f"Line Loss: {losses_line.val:.4f} ({losses_line.avg:.4f}) "
                      f"Top1: {top1.val:.3f} ({top1.avg:.3f}) ")
        plotter.plot('train loss', 'train', 'train Loss', epoch, losses.avg)

        # evaluate on test set (i know this is not fine)
        acc1= validating(val_loader, network, ce_loss, _log, device, epoch, _run._id)
        best_acc1 = max(best_acc1, acc1)
        _log.info(f"epoch: {epoch}, best accuracy: {best_acc1}")
        torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))


def validating(data_loader, network, criterion, log, device, epoch, runid):
    validate_dir = os.path.join('experiments/contact', str(runid), 'validate')
    if not os.path.exists(validate_dir):
        os.makedirs(validate_dir)
    losses = AverageMeter()
    top1 = AverageMeter()
    ioues = AverageMeter()
    line_mses = AverageMeter()
    mse_loss = torch.nn.L1Loss()
    network.eval()
    cv2.setNumThreads(0)
    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            image = sample["image"]
            target = sample["target"]
            gt_line = sample["gt_linemask_d"]
            ori_image = sample["ori_image"]

            image = image.to(device)
            target = target[:,0]
            target = target.to(device)
            gt_line = gt_line.to(device)

            output, line_prob = network(image)
            loss = criterion(output, target)

            losses.update(loss.item())
            acc1 = accuracy(output, target)
            top1.update(acc1[0].item())

            for i in range(image.size(0)):
                #if target[i].item() == 1:
                line_mse = mse_loss(line_prob[i], gt_line[i])
                iou = comp_iou(line_prob[i:i+1]>0.5, gt_line[i:i+1]>0.5)
                ioues.update(iou.item())# you should eval for contact only
                line_mses.update(line_mse.item())

                plane_mask1 = cv2.resize(image[i,3].cpu().numpy(), dsize=(640,480))
                plane_mask2 = cv2.resize(image[i,4].cpu().numpy(), dsize=(640,480))
                image0 = ori_image[i].numpy().copy().astype(np.uint8)
                image0[plane_mask1>0.5,0] = 255
                image0[plane_mask2>0.5,2] = 255

                gt_mask = cv2.resize(gt_line[i].squeeze().cpu().numpy(), dsize=(640,480))
                re_mask = cv2.resize(line_prob[i].squeeze().cpu().numpy(), dsize=(640,480))
                black_img = np.zeros((480,640,3), dtype=np.uint8)
                black_img[:,:,0] = re_mask*255
                black_img[:,:,1] = gt_mask*255
                black_img[re_mask>0.5,2] = 255
                cv2.imwrite(os.path.join(validate_dir, f"img_{i}.png"), np.concatenate([image0, black_img], 1))

        log.info(f"Loss: {losses.avg:.4f}, Top1: {top1.avg:.4f}, IOU: {ioues.avg:.1f}, MSE: {line_mses.avg:.4f}")

    return top1.avg # consider to return val...

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

    ex.add_config('./configs/config_contact.yaml')
    ex.run_commandline()
