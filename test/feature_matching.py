import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from r2d2.r2d2 import R2D2_Creator
from superpoint.superpoint import SuperPoint
from sekd.sekd import SEKD
import time

class Sift(torch.nn.Module):
    def __init__(self):
        super(Sift, self).__init__()
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)

    def forward(self,image):
        kp,desc = self.sift.detectAndCompute(image, None)
        scores = np.array([k.response for k in kp])
        return kp,scores,desc

class Orb(torch.nn.Module):
    def __init__(self):
        super(Orb, self).__init__()
        self.orb = cv2.ORB_create(nfeatures=500)

    def forward(self,image):
        kp, desc = self.orb.detectAndCompute(image,None)
        scores = np.array([k.response for k in kp])
        return kp, scores, desc


class R2D2(torch.nn.Module):
    def __init__(self):
        super(R2D2, self).__init__()
        self.r2d2 = R2D2_Creator()

    def forward(self,image):
        kp,desc,scores = self.r2d2.detectAndCompute(image)
        return kp, scores, desc

class Sekd(torch.nn.Module):
    def __init__(self):
        super(Sekd, self).__init__()
        self.sekd = SEKD(multi_scale=True,cuda=True)

    def forward(self,image):
        kp, scores, desc = self.sekd.detectAndCompute(image)
        return kp, scores, desc

class SuperpointNet(torch.nn.Module):
    def __init__(self,config):
        super(SuperpointNet, self).__init__()
        self.superpoint = SuperPoint(config.get('superpoint'))

    def forward(self,image):
        pred = self.superpoint({'image':image})
        kp = pred['keypoints'][0].cpu().detach().numpy() # nx2 numpy
        desc = pred['descriptors'][0].cpu().detach().numpy().T
        scores = pred['scores'][0].cpu().detach().numpy()
        return kp, scores, desc

class Matching(torch.nn.Module):
    def __init__(self,config={}):
        super(Matching, self).__init__()
        self.config = config
        if config.get('sift') != None:
            self.sift = Sift()
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        if config.get('orb') != None:
            self.orb = Orb()
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        if config.get('r2d2') != None:
            self.r2d2 = R2D2()
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        if config.get('superpoint') != None:
            self.superpoint = SuperpointNet(config)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        if config.get('sekd') != None:
            self.sekd = Sekd()
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        if config.get('flann') != None:# KNN based on KD-tree
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        if config.get('brute-force') != None:
            # if config.get('orb') != None:
            #     self.knn = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            # else:
            self.knn = cv2.BFMatcher()

    def forward(self,image0,image1):
        time1 = time.time()
        if self.config.get('sift') != None:
            kp0, scores0, desc0 = self.sift(image0)
            kp1, scores1, desc1 = self.sift(image1)
        if self.config.get('orb') != None:
            kp0, scores0, desc0 = self.orb(image0)
            kp1, scores1, desc1 = self.orb(image1)
        if self.config.get('r2d2') != None:
            kp0, scores0, desc0 = self.r2d2(image0)
            kp1, scores1, desc1 = self.r2d2(image1)
        if self.config.get('superpoint') != None:
            kp0, scores0, desc0 = self.superpoint(image0)
            kp1, scores1, desc1 = self.superpoint(image1)
        if self.config.get('sekd') != None:
            kp0, scores0, desc0 = self.sekd(image0)
            kp1, scores1, desc1 = self.sekd(image1)
        time2 = time.time()

        if self.config.get('flann') != None:
            matches = self.flann.knnMatch(desc0,desc1,k=2)
        if self.config.get('brute-force') != None:
            matches = self.knn.knnMatch(desc0,desc1,k=2)
        if self.config.get('superglue') != None:
            #need kp0 kp1 nx2->nx2
            #need scores0, scores1 nx1->nx1
            #need desc0, desc1 nxd->dxn
            pass
        time3 = time.time()
        mask = [-1 for _ in range(len(matches))]

        coff = 0.8
        matching_score = []
        # print(len(matches))
        # print(matches[2])
        # print(matches)
        # check if there is matches has only one candidate or no candidates
        for pair in matches:
            if len(pair) == 1:
                pair.append(pair[0])
            if len(pair) == 0:
                tmp = cv2.DMatch()
                pair.append(tmp)
                pair.append(tmp)
        for i,(m,n) in enumerate(matches):
            # print(i,m,n,m.distance)
            if m.distance < coff * n.distance:
                mask[i] = m.trainIdx
            if self.config.get('sift') != None:
                matching_score.append(1 - m.distance / 500)
            if self.config.get('orb') != None:
                matching_score.append(1 - m.distance / 100)
            if self.config.get('r2d2') != None:
                matching_score.append(m.distance)
            if self.config.get('superpoint') != None:
                matching_score.append(m.distance)
            if self.config.get('sekd') != None:
                matching_score.append(m.distance)
        mask = np.array(mask)
        matching_score = np.array(matching_score)
        # valid = mask > -1

        # keypoint object -> nx2 numpy array
        if (self.config.get('r2d2') != None) or \
                (self.config.get('superpoint') != None) or \
                (self.config.get('sekd') != None):
            kpts0 = kp0
            kpts1 = kp1
        else:
            kpts0 = np.array([[k.pt[0],k.pt[1]] for k in kp0])
            kpts1 = np.array([[k.pt[0],k.pt[1]] for k in kp1])
        time4 = time.time()
        print('total time:{},extracting time:{},matching time:{},ratio test:{}'.format(
            time4-time1, time2-time1, time3-time2, time4-time3
        ))
        return kpts0, kpts1, mask, matching_score





