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
from utils.metric import eval_iou
from utils.disp import colors_256 as colorplate
from datasets.utils import calc_points_by_param_map, writePointCloud
ex = Experiment()
ex.observers.append(FileStorageObserver.create('experiments/segmentation'))

from models_segmentation.baseline_same import Baseline as MPNNet
from datasets.plane_segmentation_dataset import PlaneDataset

@ex.command
def train(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join('experiments/segmentation', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    network = MPNNet(cfg.model)
    
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
        val_dataset, batch_size=1,
        shuffle=False, num_workers=cfg.dataset.num_workers
    )

    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(cfg.num_epochs):
        # train for one epoch
        network.train()
        tic = time.time()
        for iter, sample in enumerate(train_loader):
            image = sample["image"][0] # bx6x224x224
            planercnn_mask = sample["planercnn_mask"][0].unsqueeze(1)
            gt_mask = sample["gt_mask"][0].unsqueeze(1)
            semantic = sample["semantic"][0]
            gt_semantic = sample["gt_semantic"][0]
            matched = sample["matched"][0]
            contact_mask = sample["contact_mask"][0]
            contact_split = sample["contact_split"][0]
            has_contact = sample["has_contact"][0]
            valid_num = torch.sum(matched).item()
            if valid_num==0:
                continue

            image = image.to(device)
            planercnn_mask = planercnn_mask.to(device)
            gt_mask = gt_mask.to(device)
            semantic = semantic.to(device)
            gt_semantic = gt_semantic.to(device)
            matched = matched.to(device)
            contact_mask = contact_mask.to(device)
            contact_split = contact_split.to(device)

            plane_prob = network(image)
            ################### close to planercnn or gt
            #loss = bce_loss(plane_prob[matched], gt_mask[matched])# + bce_loss(plane_prob, gt_mask)
            # tmp1 = torch.masked_select(plane_prob, semantic)
            # tmp2 = torch.masked_select(planercnn_mask, semantic)
            tmp1 = torch.masked_select(plane_prob[matched], gt_semantic)
            tmp2 = torch.masked_select(gt_mask[matched], gt_semantic)
            loss = bce_loss(tmp1, tmp2)
            ################### uniqueness
            if plane_prob.size(0)>1:
                prob_top2, _ = torch.topk(plane_prob, 2, dim=0, sorted=False)
                prob_top2 = 2. - torch.clamp(torch.sum(prob_top2, 0),min=1.)
                loss += bce_loss(prob_top2, torch.ones_like(prob_top2))
            ################## contact loss
            loss_contact, num_contact = 0., 0.
            for i in range(contact_mask.size(0)):
                if has_contact[i].item() == 0:
                    continue
                tmp1 = torch.masked_select(plane_prob[i,0], contact_mask[i])
                tmp2 = torch.masked_select(contact_split[i], contact_mask[i])
                loss_contact += bce_loss(tmp1, tmp2)
                num_contact += 1.0
            if num_contact>0.5:
                loss += loss_contact/num_contact

            losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - tic)
            tic = time.time()

            if iter % cfg.print_interval == 0:
                _log.info(f"[{epoch:2d}][{iter:5d}/{len(train_loader):5d}] "
                      f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                      f"Loss: {losses.val:.4f} ({losses.avg:.4f}) ")

        # evaluate on test set (i know this is not fine)
        validating(val_loader, network, _log, device, epoch, _run._id)
        _log.info(f"epoch: {epoch} is done")
        if epoch % 5 == 0:
            torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))


def validating(data_loader, network, log, device, epoch, runid):
    validate_dir = os.path.join('experiments/segmentation', str(runid), 'validate')
    if not os.path.exists(validate_dir):
        os.makedirs(validate_dir)
    ioues = AverageMeter()
    mse_loss = torch.nn.L1Loss()
    network.eval()
    cv2.setNumThreads(0)
    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            image = sample["image"][0]
            planercnn_mask = sample["planercnn_mask"][0]
            ori_image = sample["ori_image"][0]
            gt_mask = sample["gt_mask"][0]
            matched = sample["matched"][0]
            contact_split = sample["contact_split"][0]

            image = image.to(device)
            plane_prob = network(image)

            plane_prob = plane_prob.squeeze()
            ori_image = ori_image.numpy()
            matched = matched.numpy()

            image_dir = os.path.join(validate_dir, str(iter))
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            planercnn_seg = np.zeros_like(ori_image)
            split_seg = np.zeros_like(ori_image)
            final_seg = np.zeros_like(ori_image)
            gt_seg = np.zeros_like(ori_image)
            for i in range(image.size(0)):
                mask1 = cv2.resize(planercnn_mask[i].numpy(), dsize=(640,480))
                mask2 = cv2.resize(plane_prob[i].cpu().numpy(), dsize=(640,480))
                csplit = cv2.resize(contact_split[i].numpy(), dsize=(640,480))
                split_seg[csplit>0.5,:] = colorplate[i+1]
                planercnn_seg[mask1>0.5,:] = colorplate[i+1]
                final_seg[mask2>0.5,:] = colorplate[i+1]
                if matched[i]>0:
                    mask3 = cv2.resize(gt_mask[i].numpy(), dsize=(640,480))
                    gt_seg[mask3>0.5,:] = colorplate[i+1]
                    iou = eval_iou(mask3>0.5, mask2>0.5)
                    ioues.update(iou)

                image1 = ori_image.copy()
                image1[:,:,0] = mask1*255
                image2 = ori_image.copy()
                image2[:,:,0] = mask2*255
                cv2.imwrite(os.path.join(image_dir, f"img_{i}.png"), np.concatenate([image1, image2],1))
                cv2.imwrite(os.path.join(image_dir, f"mask_{i}.png"), np.concatenate([mask1, mask2],1)*255)
            image1 = ori_image*0.5+planercnn_seg*0.5
            image2 = ori_image*0.5+final_seg*0.5
            image3 = ori_image*0.5+gt_seg*0.5
            image4 = ori_image*0.5+split_seg*0.5
            cv2.imwrite(os.path.join(validate_dir, f"img_{iter}.png"), np.concatenate([image3, image1, image2, image4],1))
        log.info(f"validating done, IOU: {ioues.avg:.1f}")

    return 0 # consider to return val...

@ex.command
def eval(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join('experiments', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    network = MPNNet(cfg.model)
    
    if not cfg.resume_dir == 'None':
        model_dict = torch.load(cfg.resume_dir)
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)

    val_dataset = PlaneDataset(cfg.dataset, split='test', random=False, evaluation=True)
    val_loader = data.DataLoader(
        val_dataset, batch_size=36,
        shuffle=False, num_workers=cfg.dataset.num_workers
    )
    for i in range(36):
        val_dataset[i]
    exit()
    
    network.eval()
    with torch.no_grad():
        for iter, sample in enumerate(val_loader):
            image = sample["image"] # bx6x224x224
            semantic = sample["semantic"] #bx224x224
            plane_mask_out1 = sample["plane_mask_out1"] #bx224x224
            plane_mask_out2 = sample["plane_mask_out2"] #bx224x224
            line_est_mask = sample["line_est_mask"] #bx224x224
            label_id = sample["label_id"]
            label_id = label_id[:,0]
            ori_image = sample["ori_image"]
            param_map = sample["param_map"]
            camera = sample["camera"]
            semantic_map = sample["semantic_map"]
            plane_pair = sample["plane_pair"]
            

            image = image.to(device)
            semantic = semantic.to(device)
            plane_mask_out1 = plane_mask_out1.to(device)
            plane_mask_out2 = plane_mask_out2.to(device)
            line_est_mask = line_est_mask.to(device)

            plane_prob1, plane_prob2 = network(image)

            for i in range(image.size(0)):
                if label_id[i].item() == 0:
                    continue
                mask_out1 = cv2.resize(plane_mask_out1[i].cpu().numpy(), dsize=(640,480))*255
                mask_out2 = cv2.resize(plane_mask_out2[i].cpu().numpy(), dsize=(640,480))*255
                prob1 = cv2.resize(plane_prob1[i].squeeze().cpu().numpy(), dsize=(640,480))*255
                prob2 = cv2.resize(plane_prob2[i].squeeze().cpu().numpy(), dsize=(640,480))*255
                mline = cv2.resize(line_est_mask[i].cpu().numpy(), dsize=(640,480)).astype(np.bool)
                ori_img = ori_image[i].numpy().astype(np.float32)
                imgshow1 = blend_img(ori_img, mask_out1, mask_out2, mline)
                imgshow2 = blend_img(ori_img, prob1, prob2, mline)
                
                smap = semantic_map[i].numpy().astype(np.bool)
                pmap = param_map[i].numpy()
                
                points = calc_points_by_param_map(pmap, camera[i].numpy())
                image1 = ori_img.copy()
                image1[mask_out1>125,0] = 255
                image1[mask_out2>125,2] = 255
                colors = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).reshape(-1,3)
                points = np.concatenate([points, colors],1)
                points = points[smap.reshape(-1),:]
                writePointCloud(f"./experiments/segment/{i}_blend_before.ply",points)

                smap[prob1>125] = True
                smap[prob2>125] = True
                ppair = plane_pair[i].numpy()
                pmap[prob1>125,:] = ppair[0]
                pmap[prob2>125,:] = ppair[1]
                image1 = ori_img.copy()
                image1[prob1>125,0] = 255
                image1[prob2>125,2] = 255
                points = calc_points_by_param_map(pmap, camera[i].numpy())
                colors = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).reshape(-1,3)
                points = np.concatenate([points, colors],1)
                points = points[smap.reshape(-1),:]
                writePointCloud(f"./experiments/segment/{i}_blend_after.ply",points)
                cv2.imwrite(f"./experiments/segment/{i}_blend.png", np.concatenate([imgshow1, imgshow2]))
                print(i)

def blend_img(image, mask1, mask2, line):
    i1 = image.copy()
    t=125.
    condi = mask1>t
    i1[condi,0] = mask1[condi]
    i2 = image.copy()
    condi = mask2>t
    i2[condi,2] = mask2[condi]
    i3 = image.copy()
    condi = mask1>t
    i3[condi,0] = mask1[condi]
    condi = mask2>t
    i3[condi,2] = mask2[condi]
    i4 = i3.copy()
    i4[line,:] = 255
    return np.concatenate([i1,i2,i3,i4],1)  
      
if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), \
        'PyTorch 1.0.0 is used'

    ex.add_config('./configs/config_segmentation.yaml')
    ex.run_commandline()
