"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import glob
import cv2
import os
import pickle

from datasets.utils import cleanSegmentation

class ScanNetScene():
    """ This class handle one scene of the scannet dataset and provide interface for dataloaders """
    def __init__(self, config, scenePath, scene_id, confident_labels, layout_labels,):
        self.config = config
        
        self.scannetVersion = 2
        self.confident_labels, self.layout_labels = confident_labels, layout_labels
        self.camera = np.zeros(6)

        if self.scannetVersion == 1:
            with open(scenePath + '/frames/_info.txt') as f:
                for line in f:
                    line = line.strip()
                    tokens = [token for token in line.split(' ') if token.strip() != '']
                    if tokens[0] == "m_calibrationColorIntrinsic":
                        intrinsics = np.array([float(e) for e in tokens[2:]])
                        intrinsics = intrinsics.reshape((4, 4))
                        self.camera[0] = intrinsics[0][0]
                        self.camera[1] = intrinsics[1][1]
                        self.camera[2] = intrinsics[0][2]
                        self.camera[3] = intrinsics[1][2]                    
                    elif tokens[0] == "m_colorWidth":
                        self.colorWidth = int(tokens[2])
                    elif tokens[0] == "m_colorHeight":
                        self.colorHeight = int(tokens[2])
                    elif tokens[0] == "m_depthWidth":
                        self.depthWidth = int(tokens[2])
                    elif tokens[0] == "m_depthHeight":
                        self.depthHeight = int(tokens[2])
                    elif tokens[0] == "m_depthShift":
                        self.depthShift = int(tokens[2])
                    elif tokens[0] == "m_frames.size":
                        self.numImages = int(tokens[2])
                        pass
                    continue
                pass
            self.imagePaths = glob.glob(scenePath + '/frames/frame-*color.jpg')
        else:
            with open(scenePath + '/' + scene_id + '.txt') as f:
                for line in f:
                    line = line.strip()
                    tokens = [token for token in line.split(' ') if token.strip() != '']
                    if tokens[0] == "fx_depth":
                        self.camera[0] = float(tokens[2])
                    if tokens[0] == "fy_depth":
                        self.camera[1] = float(tokens[2])
                    if tokens[0] == "mx_depth":
                        self.camera[2] = float(tokens[2])                            
                    if tokens[0] == "my_depth":
                        self.camera[3] = float(tokens[2])                            
                    elif tokens[0] == "colorWidth":
                        self.colorWidth = int(tokens[2])
                    elif tokens[0] == "colorHeight":
                        self.colorHeight = int(tokens[2])
                    elif tokens[0] == "depthWidth":
                        self.depthWidth = int(tokens[2])
                    elif tokens[0] == "depthHeight":
                        self.depthHeight = int(tokens[2])
                    elif tokens[0] == "numDepthFrames":
                        self.numImages = int(tokens[2])
                        pass
                    continue
                pass
            self.depthShift = 1000.0
            self.imagePaths = [scenePath + '/frames/color/' + str(imageIndex) + '.jpg' for imageIndex in range(self.numImages - 1)]
            pass
            
        self.camera[4] = self.depthWidth
        self.camera[5] = self.depthHeight
        self.planes = np.load(scenePath + '/annotation/planes.npy')

        self.plane_info = np.load(scenePath + '/annotation/plane_info.npy')            
        if len(self.plane_info) != len(self.planes):
            print('invalid number of plane info', scenePath + '/annotation/planes.npy', scenePath + '/annotation/plane_info.npy', len(self.plane_info), len(self.planes))
            exit(1)

        self.scenePath = scenePath
        #self.planeId_imageId = pickle.load(open(scenePath + '/planeId_imageId.pickle', 'rb'))
        #a = np.load(scenePath + '/valid_images.npy').reshape(-1)
        #b = np.arange(len(self.imagePaths))
        self.valid_image_list = np.load(scenePath + '/valid_images.npy')#b[a]
        return

    def transformPlanes(self, transformation, planes):
        planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        
        centers = planes
        centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1)
        newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
        newCenters = newCenters[:, :3] / newCenters[:, 3:4]

        refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
        refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
        newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
        newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

        planeNormals = newRefPoints - newCenters
        planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
        planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
        newPlanes = planeNormals * planeOffsets
        return newPlanes
    def my_transformPlanes(self, transformation, planes):
        planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        planeNormals = planes / np.maximum(planeOffsets, 1e-4)
        planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)

        rotation_mat = transformation[0:3,0:3]
        rotation_transpose = np.transpose(rotation_mat)
        translation_vec = transformation[0:3, 3:4]

        newNormals = np.matmul(planeNormals, rotation_transpose)
        newOffsets = np.matmul(newNormals, translation_vec) + planeOffsets
        newPlanes = newNormals * newOffsets
        return newPlanes
    def transformPlane2Global(self, transformation, planes):
        planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        planeNormals = planes / np.maximum(planeOffsets, 1e-4)
        planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)

        rotation_mat = transformation[0:3,0:3]
        translation_vec = transformation[0:3, 3:4]

        newNormals = np.matmul(planeNormals, rotation_mat)
        newOffsets = planeOffsets-np.matmul(planeNormals, translation_vec)
        #return newNormals * newOffsets
        return np.concatenate([newNormals, newOffsets],1)
    ## compute scene XYZ from a depth map
    def calcSceneXYZ(self, depth, camera, transformation):
        #depth = cv2.inpaint(depth, (depth < 1e-4).astype(np.uint8), 5, cv2.INPAINT_NS)
        height = depth.shape[0]
        width = depth.shape[1]

        urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
        vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
        ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)

        CameraXYZ = np.expand_dims(depth, -1) * ranges
        CameraXYZ = CameraXYZ.transpose(2,0,1)
        CameraXYZ = CameraXYZ.reshape(3,-1)

        rotation_mat = transformation[0:3,0:3]
        rotation_transpose = np.transpose(rotation_mat)
        translation_vec = transformation[0:3, 3:4]

        WorldXYZ = np.matmul(rotation_transpose, CameraXYZ-translation_vec)
        WorldXYZ = WorldXYZ.reshape(3, height, width)
        return WorldXYZ


    def __len__(self):
        return len(self.valid_image_list)
    
    def __getitem__(self, imageIndex):
        imagePath = self.imagePaths[imageIndex]
        image = cv2.imread(imagePath)

        if self.scannetVersion == 1:
            segmentationPath = imagePath.replace('frames/', 'annotation/segmentation/').replace('color.jpg', 'segmentation.png')
            depthPath = imagePath.replace('color.jpg', 'depth.pgm')
            posePath = imagePath.replace('color.jpg', 'pose.txt')
        else:
            segmentationPath = imagePath.replace('frames/color/', 'annotation/segmentation/').replace('.jpg', '.png')
            depthPath = imagePath.replace('color', 'depth').replace('.jpg', '.png')
            posePath = imagePath.replace('color', 'pose').replace('.jpg', '.txt')
            semanticsPath = imagePath.replace('color/', 'instance-filt/').replace('.jpg', '.png')            
            pass

        try:
            depth = cv2.imread(depthPath, -1).astype(np.float32) / self.depthShift
        except:
            print('no depth image', depthPath, self.scenePath)
            exit(1)

        extrinsics_inv = []
        with open(posePath, 'r') as f:
            for line in f:
                extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                continue
            pass
        extrinsics_inv = np.array(extrinsics_inv).reshape((4, 4))
        extrinsics = np.linalg.inv(extrinsics_inv)

        temp = extrinsics[1].copy()
        extrinsics[1] = extrinsics[2]
        extrinsics[2] = -temp

        
        segmentation = cv2.imread(segmentationPath, -1).astype(np.int32)
        
        segmentation = (segmentation[:, :, 2] * 256 * 256 + segmentation[:, :, 1] * 256 + segmentation[:, :, 0]) // 100 - 1

        segments, counts = np.unique(segmentation, return_counts=True)
        segmentList = zip(segments.tolist(), counts.tolist())
        segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]
        segmentList = sorted(segmentList, key=lambda x:-x[1])
        
        newPlanes = []
        newPlaneInfo = []
        newSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)

        newIndex = 0
        for oriIndex, count in segmentList:
            if count < self.config.planeAreaThreshold:
                continue
            if oriIndex >= len(self.planes):
                continue            
            if np.linalg.norm(self.planes[oriIndex]) < 1e-4:
                continue
            newPlanes.append(self.planes[oriIndex])
            newSegmentation[segmentation == oriIndex] = newIndex
            newPlaneInfo.append(self.plane_info[oriIndex] + [oriIndex])
            newIndex += 1
            continue

        segmentation = newSegmentation
        planes = np.array(newPlanes)
        plane_info = newPlaneInfo        
        height, width = depth.shape
        image = cv2.resize(image, (depth.shape[1], depth.shape[0]))

        if len(planes) > 0:
            #global_planes = planes
            planes = self.transformPlanes(extrinsics, planes)
            global_planes = self.transformPlane2Global(extrinsics, planes)
            segmentation, plane_depths = cleanSegmentation(image, planes, plane_info, segmentation, depth, self.camera, planeAreaThreshold=self.config.planeAreaThreshold, planeWidthThreshold=self.config.planeWidthThreshold, confident_labels=self.confident_labels, return_plane_depths=True)
            masks = (np.expand_dims(segmentation, -1) == np.arange(len(planes))).astype(np.float32)
            plane_depth = (plane_depths.transpose((1, 2, 0)) * masks).sum(2)
            plane_mask = masks.max(2)
            plane_mask *= (depth > 1e-4).astype(np.float32)            
            plane_area = plane_mask.sum()
            depth_error = (np.abs(plane_depth - depth) * plane_mask).sum() / max(plane_area, 1)
            if depth_error > 0.1:
                print('depth error', depth_error)
                planes = []
                pass
            pass
        
        if len(planes) == 0 or segmentation.max() < 0:
            print(len(planes))
            exit(1)
            pass
        try:
            p_segmentation = cv2.imread(imagePath.replace('color', 'planercnn').replace('.jpg', '_m.png'),-1)
            p_param = np.load(imagePath.replace('color', 'planercnn').replace('.jpg', '_p.npy'))
        except:
            print('no planercnn result')
            exit(1)
        p_param = p_param.reshape(-1,3)
        #p_param = self.transformPlane2Global(extrinsics, p_param)

        p_matching = np.load(imagePath.replace('color', 'planercnn').replace('.jpg', '_m.npy'))
        p_matching = p_matching.reshape(-1)

        planes = self.convert324(planes)
        p_param = self.convert324(p_param)

        info = [image, planes, plane_info, segmentation, depth, self.camera, extrinsics, p_segmentation, p_param, p_matching]
        
        return info
    def convert324(self, planes):
        plane_offsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        plane_normals = planes / np.maximum(plane_offsets, 1e-4)
        plane_normals /= np.linalg.norm(plane_normals, axis=-1, keepdims=True)
        return np.concatenate([plane_normals, plane_offsets],axis=1)
