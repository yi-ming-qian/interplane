"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf

import numpy as np
import time
import datasets.utils as utils
import os
import cv2
from PIL import Image

from datasets.scannet_scene import ScanNetScene

class PlaneDataset(Dataset):
    def __init__(self, config, split, random=True, evaluation=False):
        self.random = random
        self.dataFolder = config.dataFolder
        self.split = split
        self.eval = evaluation
        self.scenes = []
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
                if len(scene)<100:
                    continue
                self.scenes.append(scene)
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
        print(f'num of scenes {len(self.scenes)}')
        self.tmp_img_index = np.load('./datasets/img_id_list.npy').astype(np.int)

    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        
        while 1:
            if self.random:
                index = np.random.randint(len(self.scenes))
            else:
                index = index % len(self.scenes)
            if self.eval:
                tmpidx = self.tmp_img_index[index]
                index = tmpidx[0]
            # sample one frame only
            scene = self.scenes[index]
            num_scene_image = len(scene)
            image_index = np.random.randint(num_scene_image)
            image_index = scene.valid_image_list[image_index]
            if self.eval:
                image_index = tmpidx[1]
            try:
                image, planes, plane_info, segmentation, depth, camera, extrinsics, p_segmentation, p_parameter, p_matching = scene[image_index]
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
            tg_global_pids = []
            noplane_mask = np.full((480,640),True,dtype=np.bool)
            # parsing planercnn results
            for planeIndex, plane in enumerate(p_parameter):
                matched_gt_id = p_matching[planeIndex]
                if matched_gt_id >= gt_planes_num:
                    continue
                m = p_segmentation == planeIndex
                if m.sum() < 1:
                    continue
                pd_masks.append(m)
                pd_planes.append(plane)
                tg_planes.append(gt_planes[matched_gt_id])
                tg_global_pids.append(gt_global_pids[matched_gt_id])
                noplane_mask[m] = False
            if len(pd_planes) == 0:
                continue
            pd_planes = np.asarray(pd_planes)
            tg_planes = np.asarray(tg_planes)
            #self.visualize(image, pd_masks, pd_planes, camera, f'{index}_{image_index}', './experiments/test/')
            #pd_points = self.visualize(image, pd_masks, pd_planes, camera, '', '').reshape(480,640,3)
            height, width = 480, 640
            urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
            vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
            ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
            ranges = ranges.reshape(height*width, 3)
            segment_depths, segment_points = self.visualize_separate(image, pd_masks, pd_planes, camera, ranges, '', '')

            tg_normals = tg_planes[:,:3]
            num_planes = tg_normals.shape[0]
            para_lst, otho_lst, other_lst = [], [], []
            for i in range(num_planes):
                pm = pd_masks[i]
                if np.sum(pm)/(height*width)<0.01:
                    continue
                for j in range(i+1, num_planes):
                    qm = pd_masks[j]
                    if np.sum(qm)/(height*width)<0.01:
                        continue

                    tmpv = np.dot(tg_normals[i], tg_normals[j])
                    if np.abs(tmpv-1.0)<0.0152:#0.05:
                        para_lst.append([i,j])
                    elif np.abs(tmpv)<0.17:#0.05:
                        otho_lst.append([i,j])
                    else:
                        other_lst.append([i,j])
            label_id = np.random.randint(3)
            if self.eval:
                label_id = image_index%3
            if label_id == 1:
                alias_lst = para_lst
            elif label_id == 0:
                alias_lst = otho_lst
            else:
                alias_lst = other_lst
            if not alias_lst:
                index = (index+1) % len(self.scenes)
                continue
            selected = np.random.randint(len(alias_lst))
            if self.eval:
                selected = 0

            p0, p1 = alias_lst[selected]
            break
        plane_mask1 = pd_masks[p0]
        plane_mask2 = pd_masks[p1]
        plane_depth1 = segment_depths[p0]
        plane_depth2 = segment_depths[p1]
        # image[plane_mask1,0] = 255
        # image[plane_mask2,2] = 255
        # cv2.imwrite(f'./experiments/testpoint/{index}_{image_index}_{label_id}.png', image)

        image = cv2.resize(image, dsize=(224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        # use two masks
        plane_mask1 = cv2.resize(plane_mask1.astype(np.float32), dsize=(224,224))
        plane_mask1 = np.expand_dims(plane_mask1, -1)
        plane_mask2 = cv2.resize(plane_mask2.astype(np.float32), dsize=(224,224))
        plane_mask2 = np.expand_dims(plane_mask2, -1)

        plane_depth1 = cv2.resize(plane_depth1, dsize=(224,224))
        plane_depth1 = np.expand_dims(plane_depth1, -1)
        plane_depth2 = cv2.resize(plane_depth2, dsize=(224,224))
        plane_depth2 = np.expand_dims(plane_depth2, -1)
		
        dot_map = np.zeros((224,224,1))
        dot_map[:,:,:] = np.abs(np.dot(pd_planes[p0,:3], pd_planes[p1,:3]))
        tmp_tensor = np.concatenate([plane_mask1, plane_mask2, plane_depth1, plane_depth2, dot_map],-1).transpose(2,0,1)
        tmp_tensor = torch.FloatTensor(tmp_tensor)
        image = torch.cat((image, tmp_tensor), dim=0)
        return image, torch.from_numpy(np.array([label_id]).astype(np.int64))

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
        return points
        pt_colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1,3)#(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) + color_map).reshape(-1,3) / 2.0
        points = np.concatenate([points, pt_colors],1)
        points = points[semantic_map.reshape(-1),:]
        utils.writePointCloud(f'{dst_path}pt-{postfix}.ply',points)

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
        return len(self.scenes)

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
