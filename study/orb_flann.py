import numpy as np
import cv2
from matplotlib import pyplot as plt

# read two grayscale images
imageA = cv2.imread('/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/04/image_1/000002.png')
imageB = cv2.imread('/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/04/image_1/000003.png')

grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

# extract sift features
# sift = cv2.xfeatures2d.(nfeatures=500)
# kp1, desc1 = sift.detectAndCompute(imageA, None)
# print(type(kp1),(kp1[0].pt[0],kp1[0].pt[1]),type(desc1),desc1.shape)
# print(len(kp1),len(desc1))
# kp2, desc2 = sift.detectAndCompute(imageB, None)
# print(len(kp2),len(desc2))
# kpImg1 = cv2.drawKeypoints(grayA, kp1, imageA)
# kpImg2 = cv2.drawKeypoints(grayB, kp2, imageB)
# cv2.imshow('kpImg1', kpImg1)
# cv2.imshow('kpImg2', kpImg2)

orb = cv2.ORB_create(nfeatures=500)
kp1, desc1 = orb.detectAndCompute(imageA, None)
kp2, desc2 = orb.detectAndCompute(imageB, None)

kpImg1 = cv2.drawKeypoints(grayA, kp1, imageA)
kpImg2 = cv2.drawKeypoints(grayB, kp2, imageB)
cv2.imshow('kpImg1', kpImg1)
cv2.imshow('kpImg2', kpImg2)


# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(desc1,desc2,k=2)
print(type(matches))
print(len(matches))
print(matches)

matchesMask = [[0,0] for i in range(len(matches))]
myMatchesMask = [ -1 for i in range(len(matches))]
coff = 0.5
count = 0
for i,(m,n) in enumerate(matches):
    if m.distance < coff*n.distance:
        matchesMask[i] = [1,0]
        myMatchesMask[i] = m.trainIdx
        count = count+1
        print(m.queryIdx, n.queryIdx)
print(len(matchesMask))
print(matchesMask)
print(count)
print(myMatchesMask)
draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0)
resultImg = cv2.drawMatchesKnn(grayA,kp1,grayB,kp2,matches,None,**draw_params)
resultImg1 = cv2.drawMatchesKnn(imageA,kp1,grayB,kp2,matches,None,**draw_params)
plt.imshow(resultImg)
plt.show()

myMatchesMask = np.array(myMatchesMask)
valid = myMatchesMask > -1
print(valid)

cv2.waitKey()