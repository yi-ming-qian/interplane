"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf

import math
import numpy as np
from scipy import stats
import time
import datasets.utils as utils
import os
import cv2
from PIL import Image
import pickle
import imageio

from datasets.scannet_scene import ScanNetScene

class PlaneDataset(Dataset):
    def __init__(self, config, split, random=True, evaluation=False):
        self.random = random
        self.dataFolder = config.dataFolder
        self.split = split
        self.eval = evaluation
        self.scenes = []
        self.sceneImageIndices = []
        self.loadClassMap()
        planenet_scene_ids_val = np.load('datasets/scene_ids_val.npy')
        planenet_scene_ids_val = {scene_id.decode('utf-8'): True for scene_id in planenet_scene_ids_val}
        with open(self.dataFolder + '/ScanNet/Tasks/Benchmark/scannetv1_' + split + '.txt') as f:
            for line in f:
                scene_id = line.strip()
                if split == 'test':
                    ## Remove scenes which are in PlaneNet's training set for fair comparison
                    if scene_id not in planenet_scene_ids_val:
                        continue
                    pass
                scenePath = self.dataFolder + '/scans/' + scene_id
                if not os.path.exists(scenePath + '/' + scene_id + '.txt') or not os.path.exists(scenePath + '/annotation/planes.npy'):
                    continue
                scene = ScanNetScene(config, scenePath, scene_id, self.confident_labels, self.layout_labels)
                self.scenes.append(scene)
                self.sceneImageIndices += [[len(self.scenes) - 1, imageIndex] for imageIndex in range(len(scene.imagePaths))]
                continue
            pass
        self.transform = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if random:
            t = int(time.time() * 1000000)
            np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))
        else:
            np.random.seed(0)
            pass
        np.random.shuffle(self.sceneImageIndices)

        self.invalid_indices = {}

        with open(self.dataFolder + '/invalid_indices_' + split + '.txt', 'r') as f:
            for line in f:
                tokens = line.split(' ')
                if len(tokens) == 3:
                    assert(int(tokens[2]) < 10000)
                    invalid_index = int(tokens[1]) * 10000 + int(tokens[2])
                    if invalid_index not in self.invalid_indices:
                        self.invalid_indices[invalid_index] = True
                        pass
                    pass
                continue
            pass

        self.sceneImageIndices = [[sceneIndex, imageIndex] for sceneIndex, imageIndex in self.sceneImageIndices if (sceneIndex * 10000 + imageIndex) not in self.invalid_indices]

        print('num images', len(self.sceneImageIndices))
        self.save_png = False
        self.return_prcnn = False
        self.tmp_img_index = np.load('./datasets/img_id_list.npy').astype(np.int)
        self.test_ids = np.loadtxt('./datasets/test_ids.txt', dtype=np.int)

    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        if self.random:
            index = np.random.randint(len(self.sceneImageIndices))
        else:
            index = index % len(self.sceneImageIndices)
        
        sceneIndex, imageIndex = self.sceneImageIndices[index]
        scene = self.scenes[sceneIndex]
        while 1: 
            if self.random:
                index = np.random.randint(len(self.sceneImageIndices))
            else:
                index = (index + 1) % self.test_ids.shape[0]#len(self.sceneImageIndices)
                pass
        
            sceneIndex, imageIndex = self.sceneImageIndices[index]
            sceneIndex, imageIndex = self.test_ids[index]
            scene = self.scenes[sceneIndex]
            try:
                image, planes, plane_info, segmentation, depth, camera, extrinsics, p_segmentation, p_parameter, p_matching = scene[imageIndex]
            except:
                print('invalid parsing')
                continue
            # parsing gt planes
            gt_planes = []
            gt_global_pids = []
            gt_masks = []
            for planeIndex, plane in enumerate(planes):
                m = segmentation == planeIndex
                if m.sum() < 1:
                    continue
                gt_masks.append(m)
                gt_planes.append(plane)
                gt_global_pids.append(plane_info[planeIndex][-1])
            gt_planes_num = len(gt_planes)
            #self.visualize(image, gt_masks, gt_planes, camera, j)

            pd_masks = []
            pd_planes = []
            tg_planes = []
            tg_masks = []
            matched_single = []
            # parsing planercnn results
            for planeIndex, plane in enumerate(p_parameter):
                m = p_segmentation == planeIndex
                if m.sum() < 1:
                    continue
                pd_masks.append(m)
                pd_planes.append(plane)
                matched_gt_id = p_matching[planeIndex]
                if matched_gt_id >= gt_planes_num:
                    tg_planes.append(np.zeros(4))
                    tg_masks.append(np.full((480,640), False))
                    matched_single.append(False)
                else:
                    tg_planes.append(gt_planes[matched_gt_id])
                    tg_masks.append(gt_masks[matched_gt_id])
                    matched_single.append(True)########
                # matched_gt_id = p_matching[planeIndex]
                # if matched_gt_id >= gt_planes_num:
                #     continue
                # pd_masks.append(m)
                # pd_planes.append(plane)
                # tg_planes.append(gt_planes[matched_gt_id])
                # tg_global_pids.append(gt_global_pids[matched_gt_id])
                # matched_single.append(True)  
                
            if len(pd_planes) == 0:
                continue
            matched_single = np.asarray(matched_single)
            pd_planes = np.asarray(pd_planes)
            tg_planes = np.asarray(tg_planes)
            pd_points, semantic_map = self.visualize(image, pd_masks, pd_planes, camera, '', '')
            pd_points = pd_points.reshape(480,640,3)
            height, width = 480, 640
            urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
            vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
            ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
            ranges = ranges.reshape(height*width, 3)
            segment_depths, segment_points = self.visualize_separate(image, pd_masks, pd_planes, camera, ranges, '', '')
            #self.visualize(image, pd_masks, pd_planes, camera, f'{index}_{image_index}', './experiments/testpoint/')
            #return 0

            # check contact or not
            if self.save_png:
                dst_path = f"./experiments/test/{index}/"
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
            num_planes = tg_planes.shape[0]
            #points = self.visualize(image, pd_masks, tg_planes, camera, '', '')
            #points = points.reshape(height,width,3)
            # label_id = np.random.randint(2) # 1 is contat, 0 is not contact
            # if self.eval:
            #     label_id = index%2
            #print(f'label_id is {label_id}')
            gt_angle, gt_contact, gt_contactline, planepair_index, matched_pair, matched_contact = [], [], [], [], [], []
            for i in range(num_planes):
                p = tg_planes[i]
                pm = pd_masks[i]
                if np.sum(pm)/(height*width)<0.01:
                    continue
                for j in range(i+1, num_planes):
                    q = tg_planes[j]
                    qm = pd_masks[j]
                    if np.sum(qm)/(height*width)<0.01:
                        continue
                    planepair_index.append([i,j])
                    matched_pair.append(matched_single[i] & matched_single[j])
                    matched_contact.append(matched_pair[-1])
                    if matched_pair[-1] == False:
                        gt_angle.append(0)
                        gt_contact.append(False)
                        gt_contactline.append(np.full((height,width), False))
                        continue
                    # angle
                    tmpv = np.dot(p[:3], q[:3])
                    if np.abs(tmpv-1.0)<0.0152:
                        gt_angle.append(1)
                    elif np.abs(tmpv)<0.17:
                        gt_angle.append(0)
                    else:
                        gt_angle.append(2)

                    # contact
                    if abs(1.0-tmpv) < 0.05:
                        matched_contact[-1] = False
                        gt_contact.append(False)
                        gt_contactline.append(np.full((height,width), False))
                        continue
                    # intersection line
                    inormal = np.cross(p[:3], q[:3])
                    ipoint = np.array([(p[3]*q[1]-p[1]*q[3])/(p[0]*q[1]-p[1]*q[0]+1e-6), (p[3]*q[0]-p[0]*q[3])/(p[1]*q[0]-p[0]*q[1]+1e-6), 0.0])
                    dist_map = self.comp_intersect_map(ranges, inormal, ipoint)
                    heat_map = np.exp(-dist_map)
                    intersect_mask = (dist_map<0.01)
                    if self.save_png:
                        blend_img = self.create_intersect_mask(pm, qm, intersect_mask, image)
                        img_show = np.concatenate([blend_img, (self.gray2color(heat_map)*255).astype(np.uint8)], axis=1)
                    # check contact or not
                    line_mask, contact_flag = self.clip_line_segment(intersect_mask, pm, qm, 25)
                    gt_contact.append(contact_flag)
                    if contact_flag:
                        gt_contactline.append(line_mask)
                    else:
                        gt_contactline.append(np.full((height,width), False))

            # if self.return_prcnn:
            #     if label_id==1:
            #         p = pd_planes[c_list[selected][0]]
            #         q = pd_planes[c_list[selected][1]]
            #     else:
            #         p = pd_planes[nc_list[selected][0]]
            #         q = pd_planes[nc_list[selected][1]]
            #     inormal = np.cross(p[:3], q[:3])
            #     ipoint = np.array([(p[3]*q[1]-p[1]*q[3])/(p[0]*q[1]-p[1]*q[0]+1e-6), (p[3]*q[0]-p[0]*q[3])/(p[1]*q[0]-p[0]*q[1]+1e-6), 0.0])
            #     dist_map = self.comp_intersect_map(ranges, inormal, ipoint)
            #     heat_map = np.exp(-dist_map)
            #     prcnn_linemask = (dist_map<0.01).astype(np.float32)
            break
        
        ori_image = image.copy()
        image = cv2.resize(image, dsize=(224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        
        ori_pd_points = pd_points.copy()
        # pd_points = cv2.resize(pd_points, dsize=(224,224), interpolation = cv2.INTER_NEAREST)
        # semantic_map = cv2.resize(semantic_map.astype(np.float32), dsize=(224,224))
        # pd_points[semantic_map<0.5,:] = 0.0
        # pd_points = pd_points.transpose(2,0,1)

        # resize mask
        pd_masks = np.asarray(pd_masks).astype(np.float32)
        pd_masks_small = np.zeros((num_planes,224,224))
        pd_masks_small_c = np.zeros((num_planes,224,224))
        segment_depths_small = np.zeros((num_planes,224,224))
        for i in range(num_planes):
            pd_masks_small[i] = cv2.resize(pd_masks[i], dsize=(224,224))
            pd_masks_small_c[i] = self.cut_shrink_mask(pd_masks_small[i]>0.5)
            segment_depths_small[i] = cv2.resize(segment_depths[i], dsize=(224,224))
        gt_angle = np.asarray(gt_angle).astype(np.int64)
        gt_contact = np.asarray(gt_contact).astype(np.int64)
        gt_contactline = np.asarray(gt_contactline).astype(np.uint8)
        matched_pair = np.asarray(matched_pair)
        matched_contact = np.asarray(matched_contact)
        planepair_index = np.asarray(planepair_index).astype(np.int16)
        #print(pd_masks.shape, gt_angle.shape, gt_contact.shape, gt_contactline.shape, planepair_index.shape)

        sample={
            "image": image,
            #"pd_points": torch.FloatTensor(pd_points),
            "pd_planes": torch.FloatTensor(pd_planes),
            "tg_planes": torch.FloatTensor(tg_planes),
            "matched_single": torch.ByteTensor(matched_single.astype(np.uint8)),
            "pd_masks_small": torch.FloatTensor(pd_masks_small),
            "pd_masks_small_c": torch.FloatTensor(pd_masks_small_c),
            "segment_depths_small": torch.FloatTensor(segment_depths_small),
            "sensor_depth": torch.FloatTensor(depth),
            "gt_angle": torch.LongTensor(gt_angle),
            "gt_contact": torch.LongTensor(gt_contact),
            "gt_contactline": torch.FloatTensor(gt_contactline),
            "planepair_index": torch.ShortTensor(planepair_index),
            "matched_pair": torch.ByteTensor(matched_pair.astype(np.uint8)),
            "matched_contact": torch.ByteTensor(matched_contact.astype(np.uint8)),
            "camera": torch.FloatTensor(camera),
            "ori_image": torch.FloatTensor(ori_image),
            "ori_pd_points": torch.FloatTensor(ori_pd_points),
            "pd_masks": torch.FloatTensor(pd_masks),
            "tg_masks": torch.FloatTensor(np.asarray(tg_masks).astype(np.float32)),
            "ransac_masks": torch.FloatTensor(np.asarray(gt_masks).astype(np.float32)),
            "ransac_planes": torch.FloatTensor(np.asarray(gt_planes)),
            #"semantic": torch.FloatTensor(semantic_map.astype(np.float32))
            "sceneIndex": sceneIndex,
            "imageIndex": imageIndex
        }
        if self.return_prcnn:
            prcnn_linemask = cv2.resize(prcnn_linemask, dsize=(224,224))
            sample["prcnn_linemask"] = torch.FloatTensor(prcnn_linemask)
        return sample

    def cut_shrink_mask(self, mask):
        ylist, xlist = np.nonzero(mask)
        ymin = np.amin(ylist)
        ymax = np.amax(ylist)
        xmin = np.amin(xlist)
        xmax = np.amax(xlist)
        ymiddle = int(np.floor((ymin+ymax)/2.0))
        xmiddle = int(np.floor((xmin+xmax)/2.0))

        total = mask.sum()
        cuts = [mask[ymin:ymiddle,xmin:xmiddle], mask[ymin:ymiddle,xmiddle:xmax], mask[ymiddle:ymax,xmin:xmiddle], mask[ymiddle:ymax,xmiddle:xmax]]
        cuts_s = []
        for cut in cuts:
            n = cut.sum()/total*600
            h, w = cut.shape
            cut_s = np.zeros((h+10,w+10), dtype=np.uint8)
            cut_s[5:h+5,5:w+5] = cut.astype(np.uint8)
            while cut_s.sum()>n:
                cut_s = cv2.erode(cut_s, np.ones((5,5),np.uint8), iterations=1)
            cuts_s.append(cut_s[5:h+5,5:w+5])
        output = np.zeros_like(mask, dtype=np.float32)
        output[ymin:ymiddle,xmin:xmiddle], output[ymin:ymiddle,xmiddle:xmax], output[ymiddle:ymax,xmin:xmiddle], output[ymiddle:ymax,xmiddle:xmax] = cuts_s[0], cuts_s[1], cuts_s[2], cuts_s[3]
        return output
    def dilate_mask(self, mask, ksize=(5,5), iternum=5):
        kernel = np.ones(ksize,np.uint8)
        return cv2.dilate(mask,kernel,iterations = iternum)
    def clip_line_segment(self, intersect_mask, mask1, mask2, n):
        ylist, xlist = np.nonzero(intersect_mask)
        npts = len(ylist)
        if npts<=11:
            return 0, False
        A = np.asarray([xlist, ylist, np.ones(npts)])
        e_vals, e_vecs = np.linalg.eig(np.matmul(A,A.T))
        abc = e_vecs[:, np.argmin(e_vals)]
        abc = abc / np.linalg.norm(abc[:2]) # caution 0
        perpen_dir = abc[:2]
        parallel_dir = np.array([-abc[1], abc[0]])
        res = np.full(intersect_mask.shape, False)

        ydir = np.arange(-n, n+1) * perpen_dir[1]
        xdir = np.arange(-n, n+1) * perpen_dir[0]
        n2 = 2*n+1
        ylistn = np.repeat(ylist, n2) + np.tile(ydir, npts)
        ylistn = np.clip(np.floor(ylistn).astype(np.int), 0, 479)
        xlistn = np.repeat(xlist, n2) + np.tile(xdir, npts)
        xlistn = np.clip(np.floor(xlistn).astype(np.int), 0, 639)
        tmp1 = mask1[ylistn, xlistn]
        tmp2 = mask2[ylistn, xlistn]
        tmp1 = np.any(tmp1.reshape(-1, n2), axis=1)
        tmp2 = np.any(tmp2.reshape(-1, n2), axis=1)
        tmp = np.logical_and(tmp1, tmp2)
        res[ylist, xlist] = tmp
        contact_flag = (np.sum(tmp)/npts)>0.05
        #if contact_flag:
        #    ylistf=ylist[tmp]
        #    xlistf=xlist[tmp]
        #    x1=xlistf[0] - 20*perpen_dir[0]
        #    y1=ylistf[0] - 20*perpen_dir[1]
        #    x2=xlistf[0] + 20*perpen_dir[0]
        #    y2=ylistf[0] + 20*perpen_dir[1]
        #    blackimage = np.zeros((480,640,3),dtype=np.uint8)
        #    blackimage[res,0] =255 
        #    cv2.line(blackimage, (int(x1),int(y1)),(int(x2),int(y2)), (0,255,0),2)
        #    x1=xlistf[0] - 20*parallel_dir[0]
        #    y1=ylistf[0] - 20*parallel_dir[1]
        #    x2=xlistf[0] + 20*parallel_dir[0]
        #    y2=ylistf[0] + 20*parallel_dir[1]
        #    cv2.line(blackimage, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255),2)
        #    cv2.imwrite(f"./experiments/test/{np.random.randint(1000)}.png", blackimage)
        return res, contact_flag

    def get_dist_flags(self, minids, grids, n, mask):
        flag = False
        for minid in minids:
             y, x = grids[minid,:]
             if self.check_dist_proj(y,x,mask,n)>0:
                 flag = True
                 break
        return flag
    def check_dist_proj(self, y, x, mask, n):
        yleft = max(y-n,0)
        yright = min(y+n,479)
        xleft = max(x-n,0)
        xright = min(x+n,639)
        tmp = mask[yleft:yright+1, xleft:xright+1]
        return np.sum(tmp)
    def get_closest_point(self, points, p, t):
        temp = np.absolute(np.matmul(points, p[:3].reshape(-1,1)) - p[3])
        return (np.where(temp<t))[0]
        #minid = np.argmin(temp)
        #return minid, temp[minid]
    def gray2color(self, img):
        return np.stack([img,img,img],axis=-1)          
    def create_intersect_mask(self, mask1, mask2, imask, image):
        m = image.copy()
        m[mask1,0] = 255
        m[imask,1] = 255
        m[mask2,2] = 255
        return m

    def comp_intersect_map(self, ranges, idir, ipoint):
        n = ranges.shape[0]
        idir = np.tile(idir, (n,1))
        tmpcross = np.cross(ranges, idir)
        up = np.absolute(np.sum(ipoint*tmpcross, axis=1))
        down = np.linalg.norm(tmpcross, axis=1)
        d = up/down
        d = d.reshape(480,640)
        return d
        #return (d<0.01).astype(np.uint8)*255
    def comp_plane_error(self, p1, p2):
        return (np.square(p1 - p2)).mean()

    def winners_take_all(self, all_images, all_cameras, all_extrinsics, ref_planes, all_masks, all_planeids):
        print(f"num of planes = {len(ref_planes)}")
        ref_masks = all_masks[0]
        ref_planeids = all_planeids[0]
        src_planeid_map = np.full((480,640),-100,dtype=np.int)
        for i in range(len(all_masks[1])):
            m = all_masks[1][i]
            src_planeid_map[m] = all_planeids[1][i]
        ref_img = all_images[0]
        ref_camera = all_cameras[0]
        ref_extrinsic = all_extrinsics[0]
        best_planes = ref_planes.copy()
        warped_error = 0.
        # for each segment
        for segid, plane in enumerate(ref_planes):
            ref_planeid = ref_planeids[segid]
            # sampling the longititude, latitude, offset
            phi = math.atan2(plane[1], plane[0])
            theta = math.acos(plane[2])
            offset = plane[3]
            mask = ref_masks[segid]
            ynz, xnz = np.nonzero(mask)
            ref_feature = ref_img[ynz,xnz,:]
            max_ncc = -100.0
            best_plane = 0
            for p in range(-20,21,4):
                for t in range(-20,21,4):
                    for o in range(-20,21,2):
                        phi_ = phi + math.radians(p)
                        theta_ = (theta + math.radians(t)) % (math.pi+1e-6)
                        offset_ = offset + o/100.0*offset
                        tmp_plane = np.array([math.sin(theta_)*math.cos(phi_), math.sin(theta_)*math.sin(phi_), math.cos(theta_), offset_])
                        tmp_ncc = 0.
                        for viewid in range(1,len(all_images)):
                            src_img = all_images[viewid]
                            src_camera = all_cameras[viewid]
                            src_extrinsic = all_extrinsics[viewid]
                            src_feature, valid_mask, _, _ = self.warping(ynz, xnz, tmp_plane, ref_camera, ref_extrinsic, src_camera, src_extrinsic, src_img, ref_planeid, src_planeid_map)
                            if np.sum(valid_mask)>10:
                                tmp_ncc = tmp_ncc + self.comp_ncc(ref_feature, src_feature, valid_mask)
                            else:
                                tmp_ncc = tmp_ncc -1.
                        tmp_ncc = tmp_ncc/(len(all_images)-1)
                        if tmp_ncc >max_ncc:
                            max_ncc =tmp_ncc
                            best_plane = tmp_plane
            if np.abs(max_ncc+1)>1e-5:
                best_planes[segid,:] = best_plane
            warped_error = warped_error + max_ncc
            print(f"seg={segid}, max_ncc={max_ncc}")
        return best_planes, warped_error/len(ref_planes)

    def warp_all(self, all_images, all_cameras, all_extrinsics, ref_planes, all_masks, all_planeids, prefix,dst_path):
        print(f"num of planes = {len(ref_planes)}")
        ref_masks = all_masks[0]
        ref_planeids = all_planeids[0]
        src_planeid_map = np.full((480,640),-100,dtype=np.int)
        for i in range(len(all_masks[1])):
            m = all_masks[1][i]
            src_planeid_map[m] = all_planeids[1][i]
        ref_img = all_images[0]
        ref_camera = all_cameras[0]
        ref_extrinsic = all_extrinsics[0]
        warped_img = np.zeros(ref_img.shape)
        ref_img_mask = np.zeros(ref_img.shape)
        o_i_b = np.zeros((480,640), dtype=np.uint8)# image boundary
        o_d_d = np.zeros((480,640), dtype=np.uint8)# depth discon, plane instance matching
        o_n_m = np.zeros((480,640), dtype=np.uint8)# not matched at all, ncc = -1
        warped_error = 0.
        # for each segment
        for segid, plane in enumerate(ref_planes):
            ref_planeid = ref_planeids[segid]
            # sampling the longititude, latitude, offset
            phi = math.atan2(plane[1], plane[0])
            theta = math.acos(plane[2])
            offset = plane[3]
            mask = ref_masks[segid]
            ynz, xnz = np.nonzero(mask)
            ref_feature = ref_img[ynz,xnz,:]
            ref_img_mask[ynz, xnz, :] = ref_feature
            for p in range(0,1):#range(-20,20,2):
                for t in range(0,1):#range(-20,20,2):
                    for o in range(0,1):#range(-10,10,1):
                        phi_ = phi + math.radians(p)
                        theta_ = (theta + math.radians(t)) % (math.pi+1e-6)
                        offset_ = offset + o/100.0*offset
                        tmp_plane = np.array([math.sin(theta_)*math.cos(phi_), math.sin(theta_)*math.sin(phi_), math.cos(theta_), offset_])
                        for viewid in range(1,len(all_images)):
                            src_img = all_images[viewid]
                            src_camera = all_cameras[viewid]
                            src_extrinsic = all_extrinsics[viewid]
                            src_feature, valid_mask, o_i_b_, o_d_d_ = self.warping(ynz, xnz, tmp_plane, ref_camera, ref_extrinsic, src_camera, src_extrinsic, src_img, ref_planeid, src_planeid_map)
                            #valid_mask[:] = True
                            if np.sum(valid_mask)>10:
                                warped_error = warped_error + self.comp_ncc(ref_feature, src_feature, valid_mask)
                            else:
                                warped_error = warped_error -1.
                                o_n_m[ynz, xnz] = 255
                            src_feature[~valid_mask,:] = np.asarray([0,0,0])
                            warped_img[ynz, xnz, :] = src_feature
                            o_i_b[ynz, xnz] = (1 - o_i_b_.astype(np.uint8)) * 255
                            o_d_d[ynz, xnz] = (1 - o_d_d_.astype(np.uint8)) * 255

        cv2.imwrite(f'{dst_path}{prefix}-warped.png', warped_img)
        cv2.imwrite(f'{dst_path}{prefix}-ref.png', ref_img)
        cv2.imwrite(f'{dst_path}{prefix}-ref-mask.png', ref_img_mask)
        warped_img_blend = cv2.addWeighted(ref_img_mask, 0.5, warped_img, 0.5, 0.0)
        cv2.imwrite(f'{dst_path}{prefix}-warped-blend.png', warped_img_blend)
        images = [cv2.cvtColor(warped_img_blend.astype(np.uint8), cv2.COLOR_BGR2RGB), cv2.cvtColor(warped_img.astype(np.uint8), cv2.COLOR_BGR2RGB)]
        imageio.mimsave(f'{dst_path}{prefix}-warped-movie.gif', images, duration=0.5)
        cv2.imwrite(f'{dst_path}{prefix}-occ-img-boundary.png', o_i_b)
        cv2.imwrite(f'{dst_path}{prefix}-occ-depth-discon.png', o_d_d)
        cv2.imwrite(f'{dst_path}{prefix}-not-matched-atall.png', o_n_m)
        cv2.imwrite(f'{dst_path}{prefix}-src.png', all_images[-1])
        src_img_mask = np.zeros(all_images[-1].shape, dtype=np.uint8)
        for m in all_masks[-1]:
            src_img_mask[m] = all_images[-1][m]
        cv2.imwrite(f'{dst_path}{prefix}-src-mask.png', src_img_mask)
        return warped_error/(len(ref_planes))/(len(all_images)-1)
               
    def comp_ssd(self, ref_feat, src_feat, mask):
        ref_feat = ref_feat/255.
        src_feat = src_feat/255.
        ref_feat = ref_feat[mask,:]
        src_feat = src_feat[mask,:]   
        return np.sum((ref_feat - src_feat) * (ref_feat - src_feat))/(ref_feat.shape[0]*ref_feat.shape[1])  
    
    def comp_ncc(self, ref_feat, src_feat, mask):
        ref_feat = ref_feat/255.
        src_feat = src_feat/255.
        ref_feat = ref_feat[mask,:]
        src_feat = src_feat[mask,:]
        res = 0.
        for i in range(3):
            res = res + self.comp_ncc_1d(ref_feat[:,i], src_feat[:,i])
        return res/3.0
        
    def comp_ncc_1d(self, x1, x2):
        x1mean = x1-np.mean(x1)
        x2mean = x2-np.mean(x2)
        y1 = np.sqrt(np.sum(x1mean * x1mean))
        y2 = np.sqrt(np.sum(x2mean * x2mean))
        result = (x1mean * x2mean)/(y1*y2)
        return np.sum(result)

            

    def warping(self, ynz, xnz, plane, ref_camera, ref_extrinsic, src_camera, src_extrinsic, src_img, ref_planeid, src_planeid_map):
        normal = plane[0:3].reshape(3,1)
        offset = plane[3:4][0]
        # reference view
        urange = (xnz - ref_camera[2])/ref_camera[0]
        vrange = (ynz - ref_camera[3])/ref_camera[1]
        ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
        
        normalXYZ = np.matmul(ranges, normal)
        plane_depth = offset / np.maximum(normalXYZ, 1e-4)
        cameraXYZ = plane_depth * ranges
        cameraXYZ = cameraXYZ.transpose()
        
        ref_rotation = ref_extrinsic[0:3, 0:3]
        ref_translate = ref_extrinsic[0:3, 3:4]
        worldXYZ = np.matmul(np.transpose(ref_rotation), cameraXYZ-ref_translate)

        # source view
        src_rotation = src_extrinsic[0:3, 0:3]
        src_translate = src_extrinsic[0:3, 3:4]
        cameraXYZ = np.matmul(src_rotation, worldXYZ) + src_translate
        cameraXYZ = cameraXYZ.transpose()
        tmpp = np.maximum(cameraXYZ[:,1], 1e-4)
        src_x = cameraXYZ[:,0]/tmpp*src_camera[0] + src_camera[2]
        src_y = -cameraXYZ[:,2]/tmpp*src_camera[1] + src_camera[3]
        
        feature = self.bilinear_interpolate(src_img, src_x, src_y)
        img_boundary_occ = (src_x>=0) & (src_y>=0) & (src_x<640) & (src_y<480)
        
        x0 = np.floor(src_x).astype(int)
        y0 = np.floor(src_y).astype(int)
        x0 = np.clip(x0, 0, 640-1)
        y0 = np.clip(y0, 0, 480-1)
        depth_discon_occ = (src_planeid_map[y0,x0].reshape(-1) == ref_planeid)
        mask = img_boundary_occ & depth_discon_occ
        return feature, mask, img_boundary_occ, depth_discon_occ

    def bilinear_interpolate(self, im, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1)
        x1 = np.clip(x1, 0, im.shape[1]-1)
        y0 = np.clip(y0, 0, im.shape[0]-1)
        y1 = np.clip(y1, 0, im.shape[0]-1)

        Ia = im[ y0, x0, : ]
        Ib = im[ y1, x0, : ]
        Ic = im[ y0, x1, : ]
        Id = im[ y1, x1, : ]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)
        wa = wa.reshape(-1,1)
        wb = wb.reshape(-1,1)
        wc = wc.reshape(-1,1)
        wd = wd.reshape(-1,1)

        return wa*Ia + wb*Ib + wc*Ic + wd*Id

    def visualize(self, image, masks, planes, camera, postfix, dst_path):
        # visualization
        param_map = np.zeros((480,640,4))
        semantic_map = np.full((480,640),False,dtype=np.bool)
        color_map = np.full((480,640,3),0,dtype=np.uint8)
        for planeIndex, plane in enumerate(planes):
            m = masks[planeIndex]
            param_map[m,:] = plane
            semantic_map[m] = True
            color_map[m,:] = np.random.randint(255, size=3)
        points = utils.calc_points_by_param_map(param_map, camera)
        return points, semantic_map
        pt_colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1,3)#(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) + color_map).reshape(-1,3) / 2.0
        points = np.concatenate([points, pt_colors],1)
        points = points[semantic_map.reshape(-1),:]
        utils.writePointCloud(f'{dst_path}pt-{postfix}.ply',points)

        #cv2.imwrite(f"./experiments/test/pt-{imgid}-seg.png",color_map)
        #cv2.imwrite(f"./experiments/test/pt-{imgid}.png", image)
    def visualize_separate(self, image, masks, planes, camera, ranges, postfix, dst_path):
        h, w = 480, 640
        htw = h*w
        segment_depths, segment_points = [], []
        #dis_points, dis_colors = [], []
        for planeIndex, plane in enumerate(planes):
            param = np.tile(plane, (htw,4))
            offset = param[:,3:4]
            normal = param[:,0:3]
            normalXYZ = (normal*ranges).sum(1).reshape(-1,1)
            plane_depth = offset / np.maximum(normalXYZ, 1e-4)
            plane_depth = np.clip(plane_depth, 0, 10)
            points = plane_depth * ranges

            segment_depths.append(plane_depth.reshape(h,w,1).squeeze())
            segment_points.append(points.reshape(h,w,3))

        #     m = masks[planeIndex].reshape(-1)
        #     dis_points.append(points[m,:])
        #     dis_colors.append(image.reshape(-1,3)[m,:])

        # dis_points = np.concatenate(dis_points)
        # dis_colors = np.concatenate(dis_colors)
        # dis_points = np.concatenate([dis_points,dis_colors],1)
        # utils.writePointCloud(f'{dst_path}pt-{postfix}_1.ply',dis_points)
        return segment_depths, segment_points

    def __len__(self):
        return len(self.sceneImageIndices)

    def loadClassMap(self):
        classLabelMap = {}
        with open(self.dataFolder + '/scannetv2-labels.combined.tsv') as info_file:
            line_index = 0
            for line in info_file:
                if line_index > 0:
                    line = line.split('\t')
                    key = line[1].strip()
                    
                    if line[4].strip() != '':
                        label = int(line[4].strip())
                    else:
                        label = -1
                        pass
                    classLabelMap[key] = label
                    classLabelMap[key + 's'] = label
                    classLabelMap[key + 'es'] = label                                        
                    pass
                line_index += 1
                continue
            pass

        confidentClasses = {'wall': True, 
                            'floor': True,
                            'cabinet': True,
                            'bed': True,
                            'chair': False,
                            'sofa': False,
                            'table': True,
                            'door': True,
                            'window': True,
                            'bookshelf': False,
                            'picture': True,
                            'counter': True,
                            'blinds': False,
                            'desk': True,
                            'shelf': False,
                            'shelves': False,
                            'curtain': False,
                            'dresser': True,
                            'pillow': False,
                            'mirror': False,
                            'entrance': True,
                            'floor mat': True,
                            'clothes': False,
                            'ceiling': True,
                            'book': False,
                            'books': False,                      
                            'refridgerator': True,
                            'television': True, 
                            'paper': False,
                            'towel': False,
                            'shower curtain': False,
                            'box': True,
                            'whiteboard': True,
                            'person': False,
                            'night stand': True,
                            'toilet': False,
                            'sink': False,
                            'lamp': False,
                            'bathtub': False,
                            'bag': False,
                            'otherprop': False,
                            'otherstructure': False,
                            'otherfurniture': False,
                            'unannotated': False,
                            '': False
        }

        self.confident_labels = {}
        for name, confidence in confidentClasses.items():
            if confidence and name in classLabelMap:
                self.confident_labels[classLabelMap[name]] = True
                pass
            continue
        self.layout_labels = {1: True, 2: True, 22: True, 9: True}
        return