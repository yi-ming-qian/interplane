import numpy as np
import torch
import cv2
import glob
import os

def comp_iou(mask1, mask2):
    mask1 = mask1.squeeze(1)
    mask2 = mask2.squeeze(1)
    intersection = torch.sum((mask1 & mask2).float(), (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = torch.sum((mask1 | mask2).float(), (1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6) * 100
    return torch.sum(iou)/mask1.size(0)

# https://github.com/davisvideochallenge/davis/blob/master/python/lib/davis/measures/jaccard.py
def eval_iou(annotation,segmentation):
    """ Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity

    """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 100.
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)*100.

def read_annotations(sceneIndex, imageIndex):
    in_path = f'./experiments/annotation/Annotations/plane_masks/{sceneIndex}_{imageIndex}/'
    if not os.path.exists(in_path):
        return np.array([])
    names = glob.glob(in_path+'*.png')
    masks = []
    for name in names:
        masks.append((cv2.imread(name,0)>100).astype(np.float32))
    plane_masks = np.asarray(masks)
    
    in_path = f'./experiments/annotation/Annotations/line_masks/{sceneIndex}_{imageIndex}/'
    names = glob.glob(in_path+'*.png')
    masks = []
    for name in names:
        masks.append((cv2.imread(name,0)>100).astype(np.float32))
    line_mask = np.asarray(masks).max(0)
    return plane_masks, line_mask


def eval_seg_annotation(pd_masks, gt_masks, line=np.array([])):
    if line.size==0:
        valid_mask = torch.FloatTensor(gt_masks.max(0)).unsqueeze(0)#torch.FloatTensor(np.full((1,480,640),1.))
    else:
        valid_mask = torch.FloatTensor(line).unsqueeze(0)
    pd_masks =  torch.FloatTensor(pd_masks)
    gt_masks = torch.FloatTensor(gt_masks)
    res = evaluateMasksTensor(pd_masks, gt_masks, valid_mask)
    print(res)
    return np.array([res[0].item(), res[1].item(), res[2].item()])

def eval_iou_annotation(pd_masks, gt_masks, line, iou_tracker):
    non_line_mask = (line<0.5)
    for gt_m in gt_masks:
        max_iou = 0.
        max_id = -1
        area = np.sum(gt_m)/640./480.
        for j, pd_m in enumerate(pd_masks):
            iou_val = eval_iou(gt_m>0.5, pd_m>0.5)
            if iou_val>max_iou:
                max_iou = iou_val
                max_id = j
        
        if max_id>=0:
            iou_tracker[0].update(max_iou, area)
            m1 = gt_m.copy()
            m2 = pd_masks[max_id].copy()
            m1[non_line_mask] = 0.
            m2[non_line_mask] = 0.
            line_iou = eval_iou(m1>0.5, m2>0.5)
            iou_tracker[1].update(line_iou, area)



def evaluateMasksTensor(predMasks, gtMasks, valid_mask, printInfo=False):
    gtMasks = torch.cat([gtMasks, torch.clamp(1 - gtMasks.sum(0, keepdim=True), min=0)], dim=0)
    predMasks = torch.cat([predMasks, torch.clamp(1 - predMasks.sum(0, keepdim=True), min=0)], dim=0)
    intersection = (gtMasks.unsqueeze(1) * predMasks * valid_mask).sum(-1).sum(-1).float()
    union = (torch.max(gtMasks.unsqueeze(1), predMasks) * valid_mask).sum(-1).sum(-1).float()    

    N = intersection.sum()
    
    RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (N * (N - 1) / 2)
    joint = intersection / N
    marginal_2 = joint.sum(0)
    marginal_1 = joint.sum(1)
    H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
    H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

    B = (marginal_1.unsqueeze(-1) * marginal_2)
    log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
    MI = (joint * log2_quotient).sum()
    voi = H_1 + H_2 - 2 * MI

    IOU = intersection / torch.clamp(union, min=1)
    SC = ((IOU.max(-1)[0] * torch.clamp((gtMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (IOU.max(0)[0] * torch.clamp((predMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2
    info = [RI, voi, SC]
    if printInfo:
        print('mask statistics', info)
        pass
    return info
# https://github.com/art-programmer/PlaneNet/blob/master/utils.py#L2115
def eval_plane_prediction(predSegmentations, gtSegmentations, predDepths, gtDepths, threshold=0.5):
    predNumPlanes = len(np.unique(predSegmentations)) - 1
    gtNumPlanes = len(np.unique(gtSegmentations)) - 1

    if len(gtSegmentations.shape) == 2:
        gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(gtNumPlanes)).astype(np.float32)
    if len(predSegmentations.shape) == 2:
        predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(predNumPlanes)).astype(np.float32)

    planeAreas = gtSegmentations.sum(axis=(0, 1))
    intersectionMask = np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 2) > 0.5

    # depthDiffs = np.expand_dims(gtDepths, -1) - np.expand_dims(predDepths, 2)
    depthDiffs = gtDepths - predDepths
    depthDiffs = depthDiffs[:, :, np.newaxis, np.newaxis]

    intersection = np.sum((intersectionMask).astype(np.float32), axis=(0, 1))

    planeDiffs = np.abs(depthDiffs * intersectionMask).sum(axis=(0, 1)) / np.maximum(intersection, 1e-4)

    planeDiffs[intersection < 1e-4] = 1

    union = np.sum(((np.expand_dims(gtSegmentations, -1) + np.expand_dims(predSegmentations, 2)) > 0.5).astype(np.float32), axis=(0, 1))
    planeIOUs = intersection / np.maximum(union, 1e-4)

    numPredictions = int(predSegmentations.max(axis=(0, 1)).sum())

    numPixels = planeAreas.sum()

    IOUMask = (planeIOUs > threshold).astype(np.float32)
    minDiff = np.min(planeDiffs * IOUMask + 1000000 * (1 - IOUMask), axis=1)
    stride = 0.05
    pixelRecalls = []
    planeStatistics = []
    for step in range(int(0.61 / stride + 1)):
        diff = step * stride
        pixelRecalls.append(np.minimum((intersection * (planeDiffs <= diff).astype(np.float32) * IOUMask).sum(1),
                                       planeAreas).sum() / numPixels)
        planeStatistics.append(((minDiff <= diff).sum(), gtNumPlanes, numPredictions))

    return pixelRecalls, planeStatistics


#https://github.com/art-programmer/PlaneNet
def evaluateDepths(predDepths, gtDepths, validMasks, planeMasks=True, printInfo=True):
    masks = np.logical_and(np.logical_and(validMasks, planeMasks), gtDepths > 1e-4)

    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt((pow(np.log(predDepths) - np.log(gtDepths), 2) * masks).sum() / numPixels)
    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
            1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    recall = float(masks.sum()) / validMasks.sum()
    if printInfo:
        print(('evaluate', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3, recall))
        pass
    return rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3, recall


def eval_plane_and_pixel_recall_normal(segmentation, gt_segmentation, param, gt_param, threshold=0.5):
    """
    :param segmentation: label map for plane segmentation [h, w] where 20 indicate non-planar
    :param gt_segmentation: ground truth label for plane segmentation where 20 indicate non-planar
    :param threshold: value for iou
    :return: percentage of correctly predicted ground truth planes correct plane
    """
    depth_threshold_list = np.linspace(0.0, 30, 13)

    # both prediction and ground truth segmentation contains non-planar region which indicated by label 20
    # so we minus one
    plane_num = len(np.unique(segmentation)) - 1
    gt_plane_num = len(np.unique(gt_segmentation)) - 1

    # 13: 0:0.05:0.6
    plane_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))
    pixel_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))

    plane_area = 0.0

    gt_param = gt_param.reshape(20, 3)

    # check if plane is correctly predict
    for i in range(gt_plane_num):
        gt_plane = gt_segmentation == i
        plane_area += np.sum(gt_plane)

        for j in range(plane_num):
            pred_plane = segmentation == j
            iou = eval_iou(gt_plane, pred_plane)

            if iou > threshold:
                # mean degree difference over overlap region:
                gt_p = gt_param[i]
                pred_p = param[j]

                n_gt_p = gt_p / np.linalg.norm(gt_p)
                n_pred_p = pred_p / np.linalg.norm(pred_p)

                angle = np.arccos(np.clip(np.dot(n_gt_p, n_pred_p), -1.0, 1.0))
                degree = np.degrees(angle)
                depth_diff = degree

                # compare with threshold difference
                plane_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32)
                pixel_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32) * \
                      (np.sum(gt_plane * pred_plane))
                break

    pixel_recall = np.sum(pixel_recall, axis=0).reshape(1, -1) / plane_area

    return plane_recall, pixel_recall

def eval_plane_param(gt, res, masks):
    areas = masks.sum(-1).sum(-1)/640/480
    areas_sum = np.sum(areas)
    tmp = np.linalg.norm(res[:,:3], axis=-1, keepdims=True)
    res[:,:3] /= tmp
    degs = np.degrees(np.arccos(np.clip(np.sum(gt[:,:3]*res[:,:3], axis=1),-1,1)))
    offsets = np.absolute(gt[:,3]-res[:,3])
    return np.sum(areas*degs)/areas_sum, np.sum(areas*offsets)/areas_sum, areas_sum
    return np.mean(degs), np.mean(offsets)
def eval_plane_param1(gt, res, masks, image):
    areas = masks.sum(-1).sum(-1)
    areas_sum = np.sum(areas)
    tmp = np.linalg.norm(res[:,:3], axis=-1, keepdims=True)
    res[:,:3] /= tmp
    degs = np.degrees(np.arccos(np.clip(np.sum(gt[:,:3]*res[:,:3], axis=1),-1,1)))
    offsets = np.absolute(gt[:,3]-res[:,3])
    index = np.argsort(degs)
    tmp = np.zeros_like(image, dtype=np.float32)
    large_errors = np.zeros(3)
    id0 = index[-1]
    tmp[masks[id0]>0.5,0] = 255
    large_errors[0] = degs[id0]
    if len(index)>1:
        id1 = index[-2]
        tmp[masks[id1]>0.5,1] = 255
        large_errors[1] = degs[id1]
    if len(index)>2:
        id2 = index[-3]
        tmp[masks[id2]>0.5,2] = 255
        large_errors[2] = degs[id2]
    return image*0.5+tmp*0.5, large_errors

def eval_planepair_diff(pd_planes, tg_planes, planepair_index, areas):
    # tmp = np.linalg.norm(pd_planes[:,:3], axis=-1, keepdims=True)
    # pd_planes[:,:3] /= tmp
    id0 = planepair_index[:,0]
    id1 = planepair_index[:,1]
    tg_degs = np.degrees(np.arccos(np.clip(np.sum(tg_planes[id0,:3]*tg_planes[id1,:3], axis=1),-1,1)))
    pd_degs = np.degrees(np.arccos(np.clip(np.sum(pd_planes[id0,:3]*pd_planes[id1,:3], axis=1),-1,1)))
    return np.sum(np.absolute(tg_degs-pd_degs)*areas)/np.sum(areas)

###### modified from planercnn evaluation code
def evaluatePlanesTensor(masks_pred, masks_gt, depth_pred, depth_gt, printInfo=False, use_gpu=False):
    """Evaluate plane detection accuracy in terms of Average Precision"""

    masks_pred = torch.round(masks_pred)
    
    plane_areas = masks_gt.sum(dim=1).sum(dim=1)
    masks_intersection = (masks_gt.unsqueeze(1) * (masks_pred.unsqueeze(0))).float()
    intersection_areas = masks_intersection.sum(2).sum(2)

    depth_diff = torch.abs(depth_gt - depth_pred)
    depth_diff[depth_gt < 1e-4] = 0

    depths_diff = (depth_diff * masks_intersection).sum(2).sum(2) / torch.clamp(intersection_areas, min=1e-4)
    depths_diff[intersection_areas < 1e-4] = 1000000
    
    union = ((masks_gt.unsqueeze(1) + masks_pred.unsqueeze(0)) > 0.5).float().sum(2).sum(2)
    plane_IOUs = intersection_areas / torch.clamp(union, min=1e-4)

    plane_IOUs = plane_IOUs.detach().cpu().numpy()
    depths_diff = depths_diff.detach().cpu().numpy()
    plane_areas = plane_areas.detach().cpu().numpy()
    intersection_areas = intersection_areas.detach().cpu().numpy()

    num_plane_pixels = plane_areas.sum()
        
    pixel_curves = []
    plane_curves = []

    for IOU_threshold in [0.5, ]:
        IOU_mask = (plane_IOUs > IOU_threshold).astype(np.float32)
        min_diff = np.min(depths_diff * IOU_mask + 1e6 * (1 - IOU_mask), axis=1)
        stride = 0.05
        plane_recall = []
        pixel_recall = []
        for step in range(21):
            diff_threshold = step * stride
            pixel_recall.append(np.minimum((intersection_areas * ((depths_diff <= diff_threshold).astype(np.float32) * IOU_mask)).sum(1), plane_areas).sum() / num_plane_pixels)
            
            plane_recall.append(float((min_diff <= diff_threshold).sum()) / len(masks_gt))
            continue
        pixel_curves.append(pixel_recall)
        plane_curves.append(plane_recall)
        continue

    APs = []
    for diff_threshold in [0.2, 0.3, 0.6, 0.9]:
        correct_mask = np.minimum((depths_diff < diff_threshold), (plane_IOUs > 0.5))
        match_mask = np.zeros(len(correct_mask), dtype=np.bool)
        recalls = []
        precisions = []
        num_predictions = correct_mask.shape[-1]
        num_targets = (plane_areas > 0).sum()
        for rank in range(num_predictions):
            match_mask = np.maximum(match_mask, correct_mask[:, rank])
            num_matches = match_mask.sum()
            precisions.append(float(num_matches) / (rank + 1))
            recalls.append(float(num_matches) / num_targets)
            continue
        max_precision = 0.0
        prev_recall = 1.0
        AP = 0.0
        for recall, precision in zip(recalls[::-1], precisions[::-1]):
            AP += (prev_recall - recall) * max_precision
            max_precision = max(max_precision, precision)
            prev_recall = recall
            continue
        AP += prev_recall * max_precision
        APs.append(AP)
        continue    

    # detection_dict['flag'] = correct_mask.max(0)
    # input_dict['flag'] = correct_mask.max(1)
    
    if printInfo:
        print('plane statistics', correct_mask.max(-1).sum(), num_targets, num_predictions)
        pass
    return plane_curves[0], pixel_curves[0]
    #return APs + plane_curves[0] + pixel_curves[0]

def comp_conrel_iou(gt_contact, gt_contactline, pred_contactprob):
    n = gt_contactline.shape[0]
    pred_iou =np.zeros(n)
    for k in range(n):
        kernel = np.ones((3,3),np.uint8)
        if gt_contact[k]==0:
            gt_mask = np.zeros((224, 224))
        else:
            gt_mask = cv2.resize(gt_contactline[k], dsize=(224,224))
            gt_mask = cv2.dilate(gt_mask,kernel,iterations = 3)
        pred_iou[k] = eval_iou(pred_contactprob[k]>0.25, gt_mask>0.25)
    return pred_iou


def eval_relation_baseline(planepair_index, planes, masks, gt_contact, gt_contactline, pc=[], pcl=[]):
    # angle baseline
    pred_angle = []
    for ppindex in planepair_index:
        i, j = ppindex
        p = planes[i]
        q = planes[j]
        tmpv = np.dot(p[:3], q[:3])
        if np.abs(tmpv-1.0)<0.0152:
            pred_angle.append(1)
        elif np.abs(tmpv)<0.17:
            pred_angle.append(0)
        else:
            pred_angle.append(2)
    # contact baseline
    pred_contact = [] 
    pred_iou = []
    
    # for k, ppindex in enumerate(planepair_index):
    #     i, j = ppindex
    #     pm = masks[i]
    #     qm = masks[j]
    #     kernel = np.ones((5,5),np.uint8)
    #     pmd = cv2.dilate(pm,kernel,iterations = 5)
    #     qmd = cv2.dilate(qm,kernel,iterations = 5)
    #     common_area = (pmd.astype(np.bool)) & (qmd.astype(np.bool))
    #     pred_contact.append((np.sum(common_area) > 10).astype(np.uint8))
    #     if pred_contact[-1] == 0:
    #         common_area = np.zeros((224,224))
    #     else:
    #         common_area = cv2.resize(common_area.astype(np.float32), dsize=(224,224))
    #     if gt_contact[k]==0:
    #         gt_mask = np.zeros((224, 224))
    #     else:
    #         gt_mask = cv2.resize(gt_contactline[k], dsize=(224,224))
    #         kernel = np.ones((3,3),np.uint8)
    #         gt_mask = cv2.dilate(gt_mask,kernel,iterations = 3)
    #     pred_iou.append(eval_iou(common_area>0.5, gt_mask>0.5))
    # return np.array(pred_angle), np.array(pred_contact), np.array(pred_iou)

    for k, ppindex in enumerate(planepair_index):
        kernel = np.ones((3,3),np.uint8)
        pred_contact.append(pc[k])
        if gt_contact[k]==0:
            gt_mask = np.zeros((224, 224))
        else:
            gt_mask = cv2.resize(gt_contactline[k], dsize=(224,224))
            gt_mask = cv2.dilate(gt_mask,kernel,iterations = 3)
        p_mask = cv2.resize(pcl[k], dsize=(224,224))
        p_mask = cv2.dilate(p_mask,kernel,iterations = 3)
        pred_iou.append(eval_iou(p_mask>0.5, gt_mask>0.5))
    return np.array(pred_angle), np.array(pred_contact), np.array(pred_iou)


def helper():
    mask1 = torch.rand((5,1,10,10)).cuda() > 0.5
    mask2 = torch.rand((5,1,10,10)).cuda() > 0.5
    print(comp_iou(mask1, mask1))
 

if __name__ == "__main__":
    print("helper")
    helper()