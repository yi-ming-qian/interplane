import os
import cv2
import time
import random
import pickle
import trimesh
import pyrender
import scipy
import matplotlib.pyplot as plt
import triangle as tr
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
from utils.metric import evaluatePlanesTensor, eval_plane_param, eval_plane_param1, eval_planepair_diff
from utils.metric import eval_relation_baseline, comp_conrel_iou
from utils.minimize import plane_minimize, validate_gradients, get_magnitude, weighted_line_fitting_2d
from utils.disp import colors_256 as colorplate
from datasets.utils import writePointCloud, drawDepthImage, calc_points_by_depth, write_obj_file
ex = Experiment()
ex.observers.append(FileStorageObserver.create('experiments/predict'))

from models_angle.baseline_same import Baseline as AngleNet
from models_contact.baseline_same import Baseline as ContactNet
from models_segmentation.baseline_same import Baseline as SegNet
from datasets.plane_predict_dataset import PlaneDataset
from datasets.plane_predict_dataset_ae import PlaneDataset as PlaneDatasetAE

@ex.command
def eval(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    checkpoint_dir = os.path.join('experiments/predict', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    angle_net = AngleNet(cfg.model)
    contact_net = ContactNet(cfg.model)
    seg_net = SegNet(cfg.model)
    
    if not cfg.resume_angle == 'None':
        model_dict = torch.load(cfg.resume_angle)
        angle_net.load_state_dict(model_dict)
    if not cfg.resume_contact == 'None':
        model_dict = torch.load(cfg.resume_contact)
        contact_net.load_state_dict(model_dict)
    if not cfg.resume_seg == 'None':
        model_dict = torch.load(cfg.resume_seg)
        seg_net.load_state_dict(model_dict)
    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        angle_net = torch.nn.DataParallel(angle_net)
        contact_net = torch.nn.DataParallel(contact_net)
        seg_net = torch.nn.DataParallel(seg_net)
    angle_net.to(device)
    contact_net.to(device1)
    seg_net.to(device)
    if cfg.input_method == "planercnn":
        val_dataset = PlaneDataset(cfg.dataset, split='test', random=False, evaluation=True)
    elif cfg.input_method == "planeae":
        val_dataset = PlaneDatasetAE(cfg.dataset, split='test', random=False, evaluation=True)
    else:
        print('input method '+cfg.input_method+' not supported!')
        exit()
    val_loader = data.DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=cfg.dataset.num_workers
    )
    
    use_gt_relation = False
    write_relation = False
    assess_seg = False
    depth_only = True

    angle_net.eval()
    contact_net.eval()
    seg_net.eval()
    angle_accuracies = AverageMeter()
    contact_accuracies = AverageMeter()
    contact_ious = AverageMeter()
    stat_parallel = np.zeros(4) # classification precision recall
    stat_ortho = np.zeros(4)
    stat_contact = np.zeros(4)
    normal_errors = [AverageMeter(), AverageMeter(), AverageMeter()]
    normal_diff_errors = [AverageMeter(), AverageMeter(), AverageMeter()]
    depth_errors = [AverageMeter(), AverageMeter(), AverageMeter()]
    offset_errors = [AverageMeter(), AverageMeter(), AverageMeter()]
    contact_depth_errors = [AverageMeter(), AverageMeter(), AverageMeter()]
    
    with torch.no_grad():
        for iter, sample in enumerate(val_loader):
            if iter==100:
                break
            sceneIndex=sample["sceneIndex"].item()
            imageIndex=sample["imageIndex"].item()

            save_path = os.path.join('experiments/predict', str(_run._id)) + f'/results/{iter}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image = sample["image"][0] #3x224x224
            #pd_points = sample["pd_points"][0] #3x224x224
            tg_planes = sample["tg_planes"][0]
            pd_planes = sample["pd_planes"][0] #planenumx4
            matched_single = sample["matched_single"][0]
            pd_masks_small = sample["pd_masks_small"][0] #planenumx224x224
            pd_masks_small_c = sample["pd_masks_small_c"][0]
            segment_depths_small = sample["segment_depths_small"][0]
            gt_angle = sample["gt_angle"][0]
            gt_contact = sample["gt_contact"][0]
            gt_contactline = sample["gt_contactline"][0]
            matched_pair = sample["matched_pair"][0]
            matched_contact = sample["matched_contact"][0]
            planepair_index = sample["planepair_index"][0]
            ori_image = sample["ori_image"][0]
            camera = sample["camera"][0]
            ori_pd_points = sample["ori_pd_points"][0]
            pd_masks = sample["pd_masks"][0] # planenumx480x640
            ransac_masks = sample["ransac_masks"][0]
            ransac_planes = sample["ransac_planes"][0]
            sensor_depth = sample["sensor_depth"][0]
            tg_masks = sample["tg_masks"][0]

            image = image.to(device)
            #pd_points = pd_points.to(device)
            pd_masks_small = pd_masks_small.to(device)
            pd_masks_small_c = pd_masks_small_c.to(device)
            segment_depths_small = segment_depths_small.to(device)
            gt_angle = gt_angle.to(device)
            gt_contact = gt_contact.to(device1)

            # build the input to network
            input_tensor, seg_tensor=[], []
            planepair_index = planepair_index.numpy()
            planepair_num = planepair_index.shape[0]
            pd_planes = pd_planes.numpy()
            tg_planes = tg_planes.numpy()
            ori_image = ori_image.numpy().astype(np.uint8)
            ori_magnitude = get_magnitude(ori_image)
            matched_single = matched_single.numpy().astype(np.bool)
            matched_pair = matched_pair.numpy().astype(np.bool)
            matched_contact = matched_contact.numpy().astype(np.bool)
            for ppindex in planepair_index:
                p, q = ppindex
                pm = pd_masks_small[p:p+1]
                qm = pd_masks_small[q:q+1]
                pdepth = segment_depths_small[p:p+1]
                qdepth = segment_depths_small[q:q+1]
                dot_map = torch.full_like(pm, np.abs(np.dot(pd_planes[p,:3],pd_planes[q,:3])), device=device)
                if cfg.model.input_channel==8:
                    input_tensor.append(torch.cat((image, pm, qm, pdepth, qdepth, dot_map), dim=0))
                elif cfg.model.input_channel==7:
                    input_tensor.append(torch.cat((image, pm, qm, pdepth, qdepth), dim=0))
                elif cfg.model.input_channel==5:
                    input_tensor.append(torch.cat((image, pm, qm), dim=0))
                else:
                    input_tensor.append(torch.cat((pm, qm), dim=0))
            input_tensor = torch.stack(input_tensor)
            print(iter, input_tensor.size())

            # inference
            try:
                angle_prob = angle_net(input_tensor)
            except:
                half_num = int(planepair_num/2.0+0.5)
                angle_prob0 = angle_net(input_tensor[0:half_num,:,:,:])
                angle_prob1 = angle_net(input_tensor[half_num:planepair_num,:,:,:])
                angle_prob = torch.cat((angle_prob0, angle_prob1), dim=0)
            input_tensor = input_tensor.to(device1)
            try:
                contact_prob, contactline_prob = contact_net(input_tensor)
            except:
                half_num = int(planepair_num/2.0+0.5)
                contact_prob0, contactline_prob0 = contact_net(input_tensor[0:half_num,:,:,:])
                contact_prob1, contactline_prob1 = contact_net(input_tensor[half_num:planepair_num,:,:,:])
                contact_prob = torch.cat((contact_prob0, contact_prob1), dim=0)
                contactline_prob = torch.cat((contactline_prob0, contactline_prob1), dim=0)
            del input_tensor

            # relation assessment
            acc_angle, pred_angle= accuracy(angle_prob, gt_angle, angle=True)
            acc_contact, pred_contact = accuracy(contact_prob, gt_contact, angle=False)
            matched_pair_num = matched_pair.sum()
            if matched_pair_num>0:
                acc_angle, acc_contact = np.sum(acc_angle[matched_pair])*100/matched_pair_num, np.sum(acc_contact[matched_pair])*100/matched_pair_num
                angle_accuracies.update(acc_angle, matched_pair_num)
                contact_accuracies.update(acc_contact, matched_pair_num)


            camera = camera.numpy()
            ranges2d = get_ranges2d(camera)
            contactline_prob = contactline_prob.cpu().numpy().squeeze()
            pd_masks = pd_masks.numpy()
            gt_contactline = gt_contactline.numpy()
            gt_angle = gt_angle.cpu().numpy()
            gt_contact = gt_contact.cpu().numpy()

            ## precision and recall
            if matched_pair_num>0:
                #pred_angle, pred_contact = eval_relation_baseline(planepair_index, pd_planes, pd_masks)
                stat_parallel += comp_precision_recall(pred_angle[matched_pair]==1, gt_angle[matched_pair]==1)
                stat_ortho += comp_precision_recall(pred_angle[matched_pair]==0, gt_angle[matched_pair]==0)
                stat_contact += comp_precision_recall(pred_contact[matched_contact]==1, gt_contact[matched_contact]==1)
                iou_flag = (gt_contact==1)&matched_contact
                if np.sum(iou_flag)>0:
                    pred_iou = comp_conrel_iou(gt_contact, gt_contactline, contactline_prob)
                    contact_ious.update(np.mean(pred_iou[iou_flag]), np.sum(iou_flag))
            
            contact_list = []
            pair_areas = np.zeros((planepair_num,1))
            contact_line_probs = []
            for i, ppindex in enumerate(planepair_index):
                p, q = ppindex
                pm = pd_masks[p]
                qm = pd_masks[q]
                #pair_areas[i] = pm.sum() + qm.sum()
                pair_areas[i] = pm.sum()/640*qm.sum()/480
                if write_relation:
                    tmp_img = ori_image.copy()
                    tmp_img[pm>0.5,0] = 255
                    tmp_img[qm>0.5,2] = 255
                    if (gt_angle[i] if use_gt_relation else pred_angle[i]) == 1:
                        cv2.imwrite(f'{save_path}para_{i}.png', tmp_img)
                        if (gt_contact[i] if use_gt_relation else pred_contact[i]) == 1:
                            cv2.imwrite(f'{save_path}coplane_{i}.png', tmp_img)
                    elif (gt_angle[i] if use_gt_relation else pred_angle[i]) ==0:
                        cv2.imwrite(f'{save_path}ortho_{i}.png', tmp_img)

                if (gt_contact[i] if use_gt_relation else pred_contact[i]) == 0:
                    continue
                gt_mask = gt_contactline[i]
                re_mask = cv2.resize(contactline_prob[i], dsize=(640,480))
                if use_gt_relation:
                    re_mask = gt_mask

                contact_line_probs.append(re_mask)
                mask_thres = 0.5
                if pred_angle[i]==1:
                    mask_thres=0.25
                ylist, xlist = extract_line2d(re_mask, mask_thres)
                raydirs = ranges2d[ylist, xlist, :]

                contact_list.append([p,q,raydirs,i])
                if write_relation:
                    black_img = np.zeros((480,640,3), dtype=np.uint8)
                    black_img[:,:,0] = re_mask*255
                    black_img[:,:,1] = gt_mask*255
                    black_img[re_mask>mask_thres,2] = 255
                    tmp_img[re_mask>0.25,1] = 255
                    cv2.imwrite(f'{save_path}contact_{i}.png', np.concatenate([tmp_img, black_img],1))
            
            contact_line_probs = np.asarray(contact_line_probs)
            # optimization
            if use_gt_relation:
                flag_para = (matched_pair & (gt_angle==1))
                flag_ortho = (matched_pair & (gt_angle==0))
                flag_contact = (matched_contact & (gt_contact==1))
                para_list = planepair_index[flag_para,:]
                ortho_list = planepair_index[flag_ortho,:]
                para_weight = pair_areas[flag_para,:]
                ortho_weight = pair_areas[flag_ortho,:]
                contact_weight = pair_areas[flag_contact,:]
                coplane_list = planepair_index[flag_para&flag_contact,:]
                coplane_weight = pair_areas[flag_para&flag_contact,:]
            else:
                flag_para, flag_ortho, flag_contact = pred_angle==1, pred_angle==0, pred_contact==1
                para_list = planepair_index[flag_para,:]
                para_weight = pair_areas[flag_para,:]
                ortho_list = planepair_index[flag_ortho,:]
                ortho_weight = pair_areas[flag_ortho,:]
                contact_weight = pair_areas[flag_contact,:]
                coplane_list = planepair_index[flag_para&flag_contact,:]
                coplane_weight = pair_areas[flag_para&flag_contact,:]
            para_weight /= np.sum(para_weight)
            ortho_weight /= np.sum(ortho_weight)
            contact_weight /= np.sum(contact_weight)
            coplane_weight /= np.sum(coplane_weight)
            
            ori_pd_points = ori_pd_points.numpy()
            point_list = []
            for i in range(pd_masks.shape[0]):
                point_list.append(ori_pd_points[pd_masks[i]>0.5,:])
            # ---------------------------------
            # -- solve the plane parameters by optimization
            cv2.imwrite(f"{save_path}image.jpg", ori_image)
            sensor_depth = sensor_depth.numpy()
            visualize(ori_image, sensor_depth, [], camera, 'point_sensor', save_path, depthonly=depth_only, sensor=True)
            ransac_masks = ransac_masks.numpy()
            ransac_planes = ransac_planes.numpy()

            p_depth_gt = visualize(ori_image, ransac_masks, ransac_planes, camera, 'point_gt', save_path, depthonly=depth_only)
            
            seg_gt = blend_image_mask(ori_image, ransac_masks, thres=0.5)
            cv2.imwrite(f"{save_path}seg_gt.png", seg_gt)

            p_depth_planercnn = visualize(ori_image, pd_masks, pd_planes, camera, 'point_planercnn', save_path, depthonly=depth_only)
            
            seg_planercnn = blend_image_mask(ori_image, pd_masks, thres=0.5)
            cv2.imwrite(f"{save_path}seg_planercnn.png", seg_planercnn)
            alpha = np.array([1.,0.,10.,1.,1.,0.,0.])
            re_planes_angle = plane_minimize(pd_planes, point_list, para_list, para_weight, ortho_list, ortho_weight, contact_list, contact_weight, coplane_list, coplane_weight, alpha)
            # p_depth_angle = visualize(ori_image, pd_masks, re_planes_angle, camera, 'point_result_angle', save_path, depthonly=depth_only)
            
            #alpha = np.array([1.,0.,10.,1.,1.,10.,0.])# ae
            alpha = np.array([1.,10.,10.,1.,1.,1.,0.])
            re_planes = plane_minimize(pd_planes, point_list, para_list, para_weight, ortho_list, ortho_weight, contact_list, contact_weight, coplane_list, coplane_weight, alpha)
            #p_depth_contact = visualize(ori_image, pd_masks, re_planes, camera, 'point_result', save_path, depthonly=depth_only)
            
            # ---------------------------------
            # -- split the contact using optimized 3d parameters
            contact_split, line_equs, line_flags, along_line_mask = expand_masks1(re_planes, contact_list, contact_line_probs, ranges2d, pd_masks)
            seg_contact = blend_image_mask(ori_image, contact_split)
            cv2.imwrite(f"{save_path}seg_contact.png", seg_contact)
            cv2.imwrite(f"{save_path}seg_contact_line.png", along_line_mask.astype(np.uint8)*255)
            #cv2.imwrite(f"{save_path}seg_expanded_line.png", expanded_seg*0.7+line_img*0.3)

            # refine segmentation by network and contact
            image = image.repeat(pd_masks_small.size(0), 1, 1, 1)
            contact_split_small = np.zeros((contact_split.shape[0],224,224),dtype=np.float32)
            for i in range(contact_split.shape[0]):
                contact_split_small[i] = cv2.resize(contact_split[i],dsize=(224,224))
            contact_split_small = torch.cuda.FloatTensor(contact_split_small)
            input_tensor = torch.cat([image, pd_masks_small.unsqueeze(1), contact_split_small.unsqueeze(1)],dim=1)
            seg_prob_small = seg_net(input_tensor)
            del input_tensor, pd_masks_small, contact_split_small
            seg_prob_small = seg_prob_small.cpu().numpy().squeeze()
            seg_prob = np.zeros((seg_prob_small.shape[0], 480, 640))
            for i, m in enumerate(seg_prob_small):
                seg_prob[i] = cv2.resize(m, dsize=(640,480))
            seg_prob = clean_prob_mask(seg_prob)
            seg_refined = blend_image_mask(ori_image, seg_prob, thres=0.5)
            cv2.imwrite(f"{save_path}seg_refined.png", seg_refined)
            p_depth_all = visualize(ori_image, seg_prob, re_planes, camera, 'point_result_ex', save_path, depthonly=depth_only)
            
            # --------------------------------
            # -- do the evaluation
            
            # 1. evaluate depth
            tg_masks = tg_masks.numpy()
            comp_depth_error(ori_image, camera, tg_masks[matched_single], tg_planes[matched_single], 
                pd_planes[matched_single], re_planes_angle[matched_single], re_planes[matched_single], depth_errors, save_path)
            # 2. evalute normal
            comp_parameter_error(tg_planes[matched_single], pd_planes[matched_single], re_planes_angle[matched_single], re_planes[matched_single], tg_masks[matched_single], normal_errors, offset_errors)
            # 3. contact depth consistency
            flag_contact = (gt_contact==1)
            comp_contact_error(gt_contactline[flag_contact], planepair_index[flag_contact], ranges2d, pd_planes, re_planes_angle, re_planes, contact_depth_errors)
            #comp_contact_error(gt_contactline, planepair_index, ranges2d, pd_planes, re_planes_angle, re_planes, contact_depth_errors, gt_contact, gt_angle, tg_planes, pd_masks)
            
            ## sensor depth
            semantic_gt = ransac_masks.max(0)
            semantic_pd = pd_masks.max(0)
            semantic_re = seg_prob.max(0)
            
            p_depth_gt[semantic_gt<0.5] = 0.
            p_depth_planercnn[semantic_pd<0.5] = 0.
            p_depth_all[semantic_re<0.5] = 0.
            cv2.imwrite(f"{save_path}depth_sensor.png", drawDepthImage(sensor_depth))
            cv2.imwrite(f"{save_path}depth_gt.png", drawDepthImage(p_depth_gt))
            cv2.imwrite(f"{save_path}depth_prcnn.png", drawDepthImage(p_depth_planercnn))
            cv2.imwrite(f"{save_path}depth_all.png", drawDepthImage(p_depth_all))

            # evaluate pairwise angular difference
            diff_flag = matched_pair#&(gt_angle!=2)
            if np.sum(diff_flag)>0:
                diff_areas = pair_areas[diff_flag,:].reshape(-1)
                normal_diff_errors[0].update(eval_planepair_diff(pd_planes, tg_planes, planepair_index[diff_flag,:], diff_areas), np.sum(diff_areas))
                normal_diff_errors[1].update(eval_planepair_diff(re_planes_angle, tg_planes, planepair_index[diff_flag,:], diff_areas), np.sum(diff_areas))
                normal_diff_errors[2].update(eval_planepair_diff(re_planes, tg_planes, planepair_index[diff_flag,:], diff_areas), np.sum(diff_areas))

        print("-----------geometry accuracy--------------")
        print(f'normal error: planercnn {normal_errors[0].avg}, angle {normal_errors[1].avg}, all {normal_errors[2].avg}')
        print(f'offset error: planercnn {offset_errors[0].avg}, angle {offset_errors[1].avg}, all {offset_errors[2].avg}')
        print(f'depth error: planercnn {depth_errors[0].avg}, angle {depth_errors[1].avg}, all {depth_errors[2].avg}')
        print(f'contact depth error: planercnn {contact_depth_errors[0].avg}, angle {contact_depth_errors[1].avg}, all {contact_depth_errors[2].avg}')
        print(f'angular diff error: planercnn {normal_diff_errors[0].avg}, angle {normal_diff_errors[1].avg}, all {normal_diff_errors[2].avg}')

        print('\n---------relation classificaiton----------')
        print(angle_accuracies.avg, contact_accuracies.avg)
        precision, recall = stat_parallel[0]/stat_parallel[1], stat_parallel[2]/stat_parallel[3] 
        print(f'parallel precision: {precision} recall: {recall} f1score: {2*(recall*precision)/(recall+precision)}')
        precision, recall = stat_ortho[0]/stat_ortho[1], stat_ortho[2]/stat_ortho[3]
        print(f'ortho precision: {precision} recall: {recall} f1score: {2*(recall*precision)/(recall+precision)}')
        precision, recall = stat_contact[0]/stat_contact[1], stat_contact[2]/stat_contact[3]
        print(f'contact precision: {precision} recall: {recall} f1score: {2*(recall*precision)/(recall+precision)}')
        print(f'contact iou: {contact_ious.avg}')

def clean_prob_mask(seg_prob):
    label = np.argmax(seg_prob,axis=0)
    semantic = seg_prob.max(0)
    label[semantic<0.5] = -1
    seg_mask = np.zeros_like(seg_prob)
    for i in range(seg_prob.shape[0]):
        seg_mask[i] = (label==i).astype(np.float64)
    return seg_mask


def comp_depth_error(ori_image, camera, tg_masks, tg_planes, pd_planes, re_planes_angle, re_planes, depth_errors, save_path):
    depth_gt = visualize(ori_image, tg_masks, tg_planes, camera, 'point_gt', save_path, depthonly=True)
    depth_prcnn = visualize(ori_image, tg_masks, pd_planes, camera, 'point_planercnn', save_path, depthonly=True)
    depth_re_angle = visualize(ori_image, tg_masks, re_planes_angle, camera, 'point_angle', save_path, depthonly=True)
    depth_re = visualize(ori_image, tg_masks, re_planes, camera, 'point_result_ex', save_path, depthonly=True)
    flag_interest = (tg_masks.max(0)>0.5)
    area_interest = np.sum(flag_interest)/640./480.
    diff_prcnn = np.absolute(depth_gt - depth_prcnn)
    diff_re_angle = np.absolute(depth_gt - depth_re_angle)
    diff_re = np.absolute(depth_gt - depth_re)
    depth_errors[0].update(np.mean(diff_prcnn[flag_interest]), area_interest)
    depth_errors[1].update(np.mean(diff_re_angle[flag_interest]), area_interest)
    depth_errors[2].update(np.mean(diff_re[flag_interest]), area_interest)
    #return get_depth_recall(diff_prcnn[flag_interest], diff_re_angle[flag_interest], diff_re[flag_interest])
def get_depth_recall(diff_prcnn, diff_re_angle, diff_re):
    stride = 0.05
    num_pixels = len(diff_prcnn)
    recalls = np.zeros((3,21))
    for step in range(21):
        diff_threshold = step * stride
        recalls[0,step] = np.sum(diff_prcnn<diff_threshold)/num_pixels
        recalls[1,step] = np.sum(diff_re_angle<diff_threshold)/num_pixels
        recalls[2,step] = np.sum(diff_re<diff_threshold)/num_pixels
    return recalls

def comp_parameter_error(tg_planes, pd_planes, re_planes_angle, re_planes, masks, normal_errors, offset_errors):
    error_normal, error_offset, area_sum = eval_plane_param(tg_planes, pd_planes, masks)
    normal_errors[0].update(error_normal, area_sum)
    offset_errors[0].update(error_offset, area_sum)
    error_normal, error_offset, area_sum = eval_plane_param(tg_planes, re_planes_angle, masks)
    normal_errors[1].update(error_normal, area_sum)
    offset_errors[1].update(error_offset, area_sum)
    error_normal, error_offset, area_sum = eval_plane_param(tg_planes, re_planes, masks)
    normal_errors[2].update(error_normal, area_sum)
    offset_errors[2].update(error_offset, area_sum)

def comp_contact_error(gt_contactline, planepair_index, ranges2d, pd_planes, re_planes_angle, re_planes, contact_errors):
    error0, error1, error2, nums = 0.,0.,0.,0.
    for i, ppindex in enumerate(planepair_index):
        p, q = ppindex
        gt_line = gt_contactline[i]
        raydirs = ranges2d[gt_line>0.5,:]
        pixel_num = raydirs.shape[0]
        # planercnn
        p_depth = get_depth_from_single_plane(raydirs, pd_planes[p])
        q_depth = get_depth_from_single_plane(raydirs, pd_planes[q])
        contact_errors[0].update(np.mean(np.absolute(p_depth-q_depth)), pixel_num/1000.)
        error0 += np.sum(np.absolute(p_depth-q_depth))
        # angle only
        p_depth = get_depth_from_single_plane(raydirs, re_planes_angle[p])
        q_depth = get_depth_from_single_plane(raydirs, re_planes_angle[q])
        contact_errors[1].update(np.mean(np.absolute(p_depth-q_depth)), pixel_num/1000.)
        error1 += np.sum(np.absolute(p_depth-q_depth))
        # prediction
        p_depth = get_depth_from_single_plane(raydirs, re_planes[p])
        q_depth = get_depth_from_single_plane(raydirs, re_planes[q])
        contact_errors[2].update(np.mean(np.absolute(p_depth-q_depth)), pixel_num/1000.)
        error2 += np.sum(np.absolute(p_depth-q_depth))
        nums += pixel_num
    

def get_depth_from_single_plane(raydirs, plane, max_depth=10):
    normal = plane[:3].reshape(-1,1)
    offset = plane[3]
    normalXYZ = np.matmul(raydirs, normal)
    plane_depth = offset / np.maximum(normalXYZ, 1e-4)
    if max_depth > 0:
        plane_depth = np.clip(plane_depth, 0, max_depth)
    return plane_depth

def extract_line2d(mask, thres):
    ylist, xlist = np.nonzero(mask>thres)
    npts = len(ylist)
    if npts<9:
        return ylist, xlist
    A = np.asarray([xlist/640., ylist/480., np.ones(npts)])
    e_vals, e_vecs = np.linalg.eig(np.matmul(A,A.T))
    abc = e_vecs[:, np.argmin(e_vals)]

    rest = np.matmul(abc.reshape(1,3), A).reshape(-1)
    flag = (np.absolute(rest)<0.001)
    return ylist[flag], xlist[flag]

def extract_line2d_ransac(mask, magnitude):
    ylist, xlist = np.nonzero(mask>0.25)
    npts = len(ylist)
    if npts<9:
        return ylist, xlist
    weight = magnitude[ylist, xlist]
    weight /= np.sum(weight)
    flag = weighted_line_fitting_2d(xlist/640., ylist/480., weight)
    return ylist[flag], xlist[flag]

def blend_image_mask(image, masks, thres=0.5):
    segcolors = np.zeros_like(image)
    for i in range(masks.shape[0]):
        segcolors[masks[i]>thres,:] = colorplate[i+1]
    return 0.3*image + 0.7*segcolors

def accuracy1(output, target, topk=(1,), angle=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        output_soft = F.softmax(output, dim=1)
        # prob_top2, _ = torch.topk(output_soft, 2)
        # flag = ((prob_top2[:,0] - prob_top2[:,1])<0.2).view(-1,1)
        prob_top2, _ = torch.topk(output_soft, 1)
        if angle:
            flag = (prob_top2<0.0).view(-1,1)
        else:
            flag = (prob_top2<0.9).view(-1,1)

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        if angle:
            pred[flag] = 2
        else:
            pred[flag] = 0
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].view(-1).float()
        
        return correct_k.cpu().numpy(), pred.view(-1).cpu().numpy()
def accuracy(output, target, topk=(1,), angle=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].view(-1).float()
        return correct_k.cpu().numpy(), pred.view(-1).cpu().numpy()
def comp_precision_recall(pred, target):
    s2 = np.sum(pred)
    s1 = np.sum(pred[target])
    s4 = np.sum(target)
    s3 = s1
    return np.array([s1,s2,s3,s4]) #precision = s1/s2, recall = s3/s4
def get_ranges2d(camera):
    height, width = 480, 640
    urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
    vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
    return ranges

def expand_masks1(planes, contact_index, contact_line_probs, ranges2d, pd_masks):#, contact_line_2d, ori_image, save_path
    mask_thres = 0.25
    pd_masks_r = np.full(pd_masks.shape, False, dtype=np.bool)
    plane_num = pd_masks.shape[0]
    line_equs = np.zeros((plane_num, plane_num, 6))
    line_flags = np.full((plane_num, plane_num), False, dtype=np.bool)
    along_line_mask = np.full((480,640),False,dtype=np.bool)
    if len(contact_index)==0:
        return pd_masks_r.astype(np.float32), line_equs, line_flags, along_line_mask
    contact_mask = contact_line_probs.max(0)>mask_thres
    contactpixel_num = np.sum(contact_mask)
    pair_label = np.full((contactpixel_num, len(contact_index)),np.nan) # nums of contact pixels x nums of contact pairs
    ranges = ranges2d.reshape(480*640,3)

    for i, ppindex in enumerate(contact_index):
        p = ppindex[0]
        q = ppindex[1]
        pm = pd_masks[p]
        qm = pd_masks[q]
        
        re_mask = contact_line_probs[i]
        im = re_mask>mask_thres
        if im.sum()<5:
            continue

        p_plane = planes[p]
        q_plane = planes[q]
        tmpv = np.dot(p_plane[:3], q_plane[:3])
        if np.abs(tmpv-1.0)<0.0152:
            intersect_mask = im
        else:
            idir = np.cross(p_plane[:3], q_plane[:3])
            ipoint = np.array([(p_plane[3]*q_plane[1]-p_plane[1]*q_plane[3])/(p_plane[0]*q_plane[1]-p_plane[1]*q_plane[0]+1e-6), 
                (p_plane[3]*q_plane[0]-p_plane[0]*q_plane[3])/(p_plane[1]*q_plane[0]-p_plane[0]*q_plane[1]+1e-6), 0.0])
            line_flags[p,q] = True
            line_flags[q,p] = True
            line_equs[p,q,:] = np.concatenate([ipoint, idir])
            line_equs[q,p,:] = np.concatenate([ipoint, idir])
            n = ranges.shape[0]
            idir = np.tile(idir, (n,1))
            tmpcross = np.cross(ranges, idir)
            up = np.absolute(np.sum(ipoint*tmpcross, axis=1))
            down = np.linalg.norm(tmpcross, axis=1)
            d = up/down
            d = d.reshape(480,640)
            intersect_mask = (d<0.01)
        
        ylist, xlist = np.nonzero(intersect_mask)
        npts = len(ylist)
        if npts<=11:
            continue
        A = np.asarray([xlist/640., ylist/480., np.ones(npts)])
        e_vals, e_vecs = np.linalg.eig(np.matmul(A,A.T))
        abc = e_vecs[:, np.argmin(e_vals)]

        if pm.sum()<qm.sum():
            selected = p
            noselected = q
        else:
            selected = q
            noselected = p

        ylist, xlist = np.nonzero(pd_masks[selected]>0.5)
        A = np.asarray([xlist/640, ylist/480, np.ones_like(xlist)])
        rest = np.matmul(abc.reshape(1,3), A).reshape(-1)
        s_tmp = rest>0.0
        s_itmp = np.invert(s_tmp)

        ylist, xlist = np.nonzero(im)
        A = np.asarray([xlist/640., ylist/480., np.ones_like(xlist)])
        rest = np.matmul(abc.reshape(1,3), A).reshape(-1)
        tmp = rest>0.0
        itmp = np.invert(tmp)
        tmp_1 = np.absolute(rest)<0.01
        along_line_mask[ylist[tmp_1], xlist[tmp_1]] = True
        # if np.sum(tmp)==0 or np.sum(itmp)==0:
        #     line_flags[p,q] = False
        #     line_flags[q,p] = False
        tmp_label = np.full((480,640),np.nan)
        if s_tmp.sum()>s_itmp.sum():
            tmp_label[ylist[tmp], xlist[tmp]] = selected
            tmp_label[ylist[itmp], xlist[itmp]] = noselected
        else:
            tmp_label[ylist[itmp], xlist[itmp]] = selected
            tmp_label[ylist[tmp], xlist[tmp]] = noselected
        pair_label[:,i] = tmp_label[contact_mask]
    labelid,_ = scipy.stats.mode(pair_label, axis=1)
    labelid = labelid.reshape(-1)
    for i in range(pd_masks_r.shape[0]):
        pd_masks_r[i,contact_mask] = (labelid==i)
    return pd_masks_r.astype(np.float32), line_equs, line_flags, along_line_mask


def visualize(image, masks, planes, camera, postfix, sceneid, depthonly=False, sensor=False):
    if sensor==False:
        param_map = np.zeros((480,640,4))
        semantic_map = np.full((480,640),False,dtype=np.bool)
        color_map = np.full((480,640,3),0,dtype=np.uint8)
        for planeIndex, plane in enumerate(planes):
            m = masks[planeIndex] > 0.5
            param_map[m,:] = plane
            semantic_map[m] = True
            color_map[m,:] = np.random.randint(255, size=3)
        points, depth = calc_dp_by_param_map(param_map, camera)
    else:
        semantic_map = masks>1e-4 # masks become depth
        points = calc_points_by_depth(masks, camera)
        depth = np.zeros((480,640))
    if depthonly:
        return depth.reshape(480,640)

    pt_colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1,3)#(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) + color_map).reshape(-1,3) / 2.0
    points = np.concatenate([points, pt_colors],1)
    points = points[semantic_map.reshape(-1),:]
    filename = f'{sceneid}{postfix}'
    writePointCloud(filename + '.ply',points)
    #return depth.reshape(480,640)

    m = pyrender.Mesh.from_points(points[:,:3], colors=points[:,3:6]/255.0)
    scene = pyrender.Scene()
    scene.add(m)
    v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True, 
            record=True, rotate=True, rotate_axis=[0,0,-1], rotate_rate = np.pi/4.0, refresh_rate=10.0)
    v.on_mouse_scroll(0,0,5,5)
    time.sleep(8)
    v.close_external()
    v.save_gif(filename+'.gif')
    return depth.reshape(480,640)

def calc_dp_by_param_map(param, camera, max_depth=10):
    if param.shape[0] == 4:
        param=param.transpose(1,2,0)
    height = param.shape[0]
    width = param.shape[1]
    urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
    vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)

    param = param.reshape(height*width, 4)
    offset = param[:,3:4]
    normal = param[:,0:3]
    #offset = np.linalg.norm(param, axis=-1, keepdims=True)
    #normal = param /  np.maximum(offset, 1e-4)

    ranges = ranges.reshape(height*width, 3)
    normalXYZ = (normal*ranges).sum(1).reshape(-1,1)
    plane_depth = offset / np.maximum(normalXYZ, 1e-4)
    if max_depth > 0:
        plane_depth = np.clip(plane_depth, 0, max_depth)
    CameraXYZ = plane_depth * ranges
    return CameraXYZ, plane_depth

def extract_polygon(image, masks, id_global):
    radius = 5
    # id_global = np.arange(480*640)+1
    # id_global = id_global.reshape(480,640)
    faces = []
    contours = []
    gn = 1
    for p in range(masks.shape[0]):
        m = masks[p]>0.5
        if np.sum(m)<50:
            continue
        # black_img = image.copy()
        # black_img[m,1] = 255  
        _, c, _= cv2.findContours(m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for k in range(len(c)):
            conto = c[k].squeeze(1)
            
            if conto.shape[0]<=3:
                continue
            if np.sum(m[conto[:,1], conto[:,0]]) != conto.shape[0]:
                print('contour pixel is not plane')
            tmp_img = image.copy()
            last_pt = np.array([-100,-100])
            r_conto = [] # downsampled conto
            for i in range(conto.shape[0]):
                x, y = conto[i]
                if x>last_pt[0]-radius and  x<last_pt[0]+radius and y>last_pt[1]-radius and y<last_pt[1]+radius:
                    continue
                if i == conto.shape[0]-1:
                    if x>conto[0,0]-radius and  x<conto[0,0]+radius and y>conto[0,1]-radius and y<conto[0,1]+radius:
                        continue
                last_pt = conto[i]
                r_conto.append(conto[i])
                #tmp_img[y,x] = np.array([0,255,0])
                #cv2.imwrite(f'./experiments/{i}.png', tmp_img)
            if len(r_conto)<=10:
                continue
            # build delauny
            r_conto = np.asarray(r_conto)
            contours.append(r_conto)
            r_gids = id_global[r_conto[:,1], r_conto[:,0]]
            # black_img[r_conto[:,1], r_conto[:,0],:] = np.random.randint(255, size=3)
            # cv2.imwrite(f'./experiments/{p}_{k}.png', black_img)
            n = r_conto.shape[0]
            seg = np.stack([np.arange(n), np.arange(n) + 1], axis=1) % n
            A = dict(vertices=r_conto, segments=seg)
            B = tr.triangulate(A, 'p')
            # tr.compare(plt, A, B)
            # plt.show()
            local_tri = B['triangles']
            flag = np.all(local_tri<n, axis=1)
            local_tri = local_tri[flag,:]
            global_tri = local_tri+gn#r_gids[local_tri.reshape(-1)].reshape(-1,3)#
            gn += n
            faces.append(global_tri)
    faces = np.concatenate(faces,axis=0)
    contours = np.concatenate(contours, axis=0)
    return faces, contours

def modify_points(points, contours, planeid_map, line_equs, line_flags, along_line_mask, planes):
    height, width = 480,640
    radius = 5
    points = points.reshape(height,width,3)
    for c in contours:
        x, y = c
        if along_line_mask[y,x]==False:
            continue
        myid = planeid_map[y,x]
        pt = points[y,x,:]
        ymin = max(0,y-radius)
        xmin = max(0,x-radius)
        ymax = min(height,y+radius+1)
        xmax = min(width,x+radius+1)
        t_pid = planeid_map[ymin:ymax,xmin:xmax]
        t_pid = np.unique(t_pid)
        pids = t_pid[t_pid>=0]
        if len(pids)<=1:
            continue
        mindist = np.inf
        minpt = np.array([])
        if len(pids)==3:# and (line_flags[pids[0],pids[1]]+line_flags[pids[0],pids[2]]+line_flags[pids[2],pids[1]])==3:
            pl = planes[pids,:]
            minpt = np.linalg.solve(planes[pids,:3], planes[pids,3:4])
            minpt = minpt.reshape(-1)
            if np.linalg.norm(minpt-pt)>=0.05:
                minpt=np.array([])
        else:
            for pid in pids:
                if pid==myid:
                    continue
                if line_flags[myid,pid]==False:
                    continue
                dist, ipt = point2line(pt, line_equs[myid,pid])
                if dist<mindist and dist<0.1:
                    mindist = dist
                    minpt = ipt
        
        if len(minpt)>0:
            points[y,x,:] = minpt
    return points.reshape(-1,3)

def point2line(pt, line):
    # tmp = np.cross(pt-line[:3], line[3:6])
    # up = np.linalg.norm(tmp)
    # down = np.linalg.norm(line[3:6])
    # distance1 = up/down # used to verify
    up = np.dot((pt-line[:3]), line[3:6])
    down = np.dot(line[3:6], line[3:6])
    t = up/down
    ipt = line[:3] + t*line[3:6]
    distance = np.linalg.norm(ipt-pt)
    return distance, ipt

def visualize_mesh(image, masks, planes, camera, postfix, sceneid, line_equs, line_flags, split_semantic, along_line_mask, depthonly=False):
    height, width = 480, 640
    param_map = np.zeros((height,width,4))
    semantic_map = np.full((height,width),False,dtype=np.bool)
    planeid_map = np.full((height,width),-1,dtype=np.int)
    for planeIndex, plane in enumerate(planes):
        m = masks[planeIndex] > 0.5
        param_map[m,:] = plane
        semantic_map[m] = True
        planeid_map[m] = planeIndex
    #planeid_map[split_semantic<0.5] = -1
    points, depth = calc_dp_by_param_map(param_map, camera)

    valid_pixel_num = np.sum(semantic_map)# do not write those zeros
    id_global = np.zeros((height, width), dtype=np.int)
    id_global[semantic_map] = np.arange(valid_pixel_num)+1
    faces, contours = extract_polygon(image, masks, id_global)

    urange = np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / width
    vrange = np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / height
    urange = urange.reshape(-1,1)
    vrange = 1-vrange.reshape(-1,1)
    uvrange = np.concatenate([urange, vrange], axis=-1)

    normal = param_map[:,:,:3]
    normal = normal.reshape(-1,3)

    semantic1d = np.arange(height*width).reshape(height,width)
    semantic1d = semantic1d[contours[:,1],contours[:,0]]
    #semantic1d = semantic_map.reshape(-1)
    filename = f'{sceneid}{postfix}'
    faces = duplicate_face(faces)
    write_obj_file(filename+'.obj', points[semantic1d,:], normal[semantic1d,:], uvrange[semantic1d,:], faces)
    if type(line_equs) == type([]):
        return 0
    new_points = modify_points(points.copy(), contours, planeid_map, line_equs, line_flags, along_line_mask, planes)
    filename = f'{sceneid}{postfix}'
    write_obj_file(filename+'_f.obj', new_points[semantic1d,:], normal[semantic1d,:], uvrange[semantic1d,:], faces)
    exit()
    return 0

    pt_colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1,3)#(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) + color_map).reshape(-1,3) / 2.0
    points = np.concatenate([points, pt_colors],1)
    points = points[semantic_map.reshape(-1),:]
    
    writePointCloud(filename + '.ply',points)

def duplicate_face(faces_ori):
    faces = faces_ori.copy()
    a = faces[:,0].copy()
    faces[:,0] = faces[:,1]
    faces[:,1] = a
    return np.concatenate([faces, faces_ori])

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), \
        'PyTorch 1.0.0 is used'

    ex.add_config('./configs/config_predict.yaml')
    ex.run_commandline()
