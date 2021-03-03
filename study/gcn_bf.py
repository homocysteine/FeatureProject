import numpy as np
import cv2
from matplotlib import pyplot as plt
from gcn.gcn import GCNv2
from utils import frame2tensor
import torch

# set cuda device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# read grayscale image
frame0 = cv2.imread('/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/04/image_1/000002.png',0)
frame1 = cv2.imread('/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/04/image_1/000003.png',0)
# from numpy to tensor
tensor0 = frame2tensor(frame0, device)
tensor1 = frame2tensor(frame1, device)

gcn = GCNv2()
desc0, det0 = gcn(tensor0)
desc1, det1 = gcn(tensor1)


