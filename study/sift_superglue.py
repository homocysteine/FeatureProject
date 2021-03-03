import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

imageA = cv2.imread('/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/04/image_1/000002.png')
imageB = cv2.imread('/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/04/image_1/000003.png')

grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

tensor1 = frame2tensor(grayA,'cuda')
tensor2 = frame2tensor(grayB, 'cuda')

# extract sift features
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
kp1, desc1 = sift.detectAndCompute(grayA, None)
kp2, desc2 = sift.detectAndCompute(grayB, None)

desc1 = torch.from_numpy(desc1.T)
desc2 = torch.from_numpy(desc2.T)

kpts1 = np.array([[k.pt[0],k.pt[1]] for k in kp1])
kpts2 = np.array([[k.pt[0],k.pt[1]] for k in kp2])
kpts1 = torch.from_numpy(kpts1)
kpts2 = torch.from_numpy(kpts2)

scores1 = np.array([k.response for k in kp1])
scores2 = np.array([k.response for k in kp2])
scores1 = torch.from_numpy(scores1)
scores2 = torch.from_numpy(scores2)



data = {
    'image0': tensor1,
    'keypoints0': kpts1,
    'scores0': scores1,
    'descriptors0': desc1,
    'image1': tensor2,
    'keypoints1': kpts2,
    'scores1': scores2,
    'descriptors1': desc2
}

superglue = torch.load('/home/yushichen/PycharmProjects/pythonProject/model/superglue_20_25000.pth')
superglue = superglue.eval.to('cuda')
pred = superglue(data)

