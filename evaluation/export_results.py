# Copyright (c) Alibaba Inc. All rights reserved.

import extract_feature

# image_dir: resized Hpatches data
# feature_dir: store the npy files of keypoints,descriptors,scores
# weights_path: DNN model files
config = {
    'image_dir':'/home/yushichen/PycharmProjects/pythonProject/data/hpatches-dataset/hpatches-sequences-resize/',
    'feature_dir':'/media/yushichen/DATA/Datasets/MyPlatformFeature/feature/',
    'weights_path':{'sekd':'/home/yushichen/PycharmProjects/pythonProject/model/sekd.pth',
                    'superpoint':'/home/yushichen/PycharmProjects/pythonProject/model/superpoint_v1.pth'},
    'max_keypoints':500,
    'nms_radius':4,
    'conf_thresh':0.55,
    'cuda':True
}

if __name__ == '__main__':
     extract_feature.extract_sekd(config=config,version=0) #single scale
     extract_feature.extract_sekd(config=config,version=1) #multi scale
     extract_feature.extract_superpoint(config=config) # superpoint
     extract_feature.extract_opencv_features(config=config,method_name='sift')
     extract_feature.extract_opencv_features(config=config,method_name='orb')
     extract_feature.extract_r2d2(config=config, version=0)
     extract_feature.extract_r2d2(config=config, version=1)

