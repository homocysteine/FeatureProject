import cv2
import numpy as np
import os
import torch
import torch.nn.init
from superpoint.superpoint import SuperPoint
from sekd.sekd import SEKD
from r2d2.patchnet import Quad_L2Net_ConfCFS
import r2d2.r2d2_extraction as r2d2
from PIL import Image
import torchvision.transforms as tvf
from r2d2.r2d2 import r2d2_val

def extract_opencv_features(config={}, method_name=''):
    print('Export {0} local features.'.format(method_name))
    # Calculate the detection result for each image in hpatches dataset.
    # For each sequence in hpatches dataset.
    if method_name == 'kaze':
        feature_extractor = cv2.KAZE_create()
    elif method_name == 'sift':
        feature_extractor = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    elif method_name == 'surf':
        feature_extractor = cv2.xfeatures2d.SURF_create()
    elif method_name == 'akaze':
        feature_extractor = cv2.AKAZE_create()
    elif method_name == 'brisk':
        feature_extractor = cv2.BRISK_create()
    elif method_name == 'orb':
        feature_extractor = cv2.ORB_create(nfeatures=500)
    else:
        print('Unknown method: ' + method_name)
        return
    for seq_name in sorted(os.listdir(config['image_dir'])):
        seq_path = os.path.join(config['image_dir'], seq_name)
        for img_name in os.listdir(seq_path):
            if img_name[-4:] != '.ppm':
                continue
            img_path = os.path.join(seq_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape
            keypoints_list, descriptors = feature_extractor.detectAndCompute(img, None)

            keypoints = np.zeros([len(keypoints_list), 2])
            scores = np.zeros(len(keypoints_list))

            for i in range(len(keypoints_list)):
                keypoints[i, 0] = keypoints_list[i].pt[0]
                keypoints[i, 1] = keypoints_list[i].pt[1]
                scores[i] = keypoints_list[i].response

            inds = np.argsort(scores)
            keypoints = keypoints[inds[:-501:-1], :]
            scores = scores[inds[:-501:-1]]

            descriptors = descriptors[inds[:-501:-1], :]
            if descriptors.dtype == np.uint8:
                dim_descriptor = descriptors.shape[1] * 8
                descriptors_float = np.zeros([keypoints.shape[0], dim_descriptor])
                for i in range(keypoints.shape[0]):
                    for j in range(dim_descriptor):
                        descriptors_float[i][j] = bool(descriptors[i][int(j/8)] & (1 << j%8))
                descriptors = descriptors_float
            print('keypoint: ', keypoints.shape)

            descriptors = descriptors.astype(np.float)
            det_dir = os.path.join(config['feature_dir'], seq_name)
            if not os.path.isdir(det_dir):
                os.makedirs(det_dir)
            det_path = os.path.join(det_dir, img_name + '.' + method_name)
            print(det_path)
            with open(det_path, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints, scores=scores, descriptors=descriptors)


def extract_superpoint(config={}):
    print('Export detection results on hpatches.')
    # model = SuperPointNet()
    outdir = config['feature_dir']
    model = SuperPoint(config={'max_keypoints':config['max_keypoints']})
    model_desc_dict = torch.load(config['weights_path']['superpoint'])
    model.load_state_dict(model_desc_dict, strict=False)
    if config['cuda']:
        model.cuda(0)
    # Calculate the detection result for each image in hpatches dataset.
    # For each sequence in hpatches dataset.
    for seq_name in os.listdir(config['image_dir']):
        seq_path = os.path.join(config['image_dir'], seq_name)
        out_path = os.path.join(outdir,seq_name)
        for img_name in os.listdir(seq_path):
            if img_name[-4:] != '.ppm':
                continue
            torch.cuda.empty_cache()
            img_path = os.path.join(seq_path, img_name)
            det_path = os.path.join(out_path, img_name + '.superpoint')
            print(det_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape
            img_height = int(img_height / 8) * 8
            img_width = int(img_width / 8) * 8
            img = img[0:img_height, 0:img_width]
            img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            if config['cuda']:
                img = img.cuda(0).float()
            img = img / 255.
            with torch.no_grad():
                res = model.forward({'image':img})
                keypoints = res['keypoints']
                scores = res['scores']
                descriptors = res['descriptors']

                #from tensor to numpy
                keypoints = keypoints[0].detach().cpu().numpy()
                scores = scores[0].detach().cpu().numpy()
                descriptors = descriptors[0].t().detach().cpu().numpy()

            print(type(keypoints),keypoints.shape)
            print(type(scores), scores.shape)
            print(type(descriptors),descriptors.shape)
            with open(det_path, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints, scores=scores, descriptors=descriptors)


def extract_sekd(config={},version=0):
    if version == 0:
        print('Export Single scale SEKD local features.')
        feature_extractor = SEKD(weights_path=config['weights_path']['sekd'],
            confidence_threshold = config['conf_thresh'], nms_radius = config['nms_radius'],
            max_keypoints = config['max_keypoints'], multi_scale=False,cuda = config['cuda'])
    elif version == 1:
        print('Export Multi scale SEKD local features.')
        feature_extractor = SEKD(weights_path=config['weights_path']['sekd'],
                                 confidence_threshold=config['conf_thresh'], nms_radius=config['nms_radius'],
                                 max_keypoints=config['max_keypoints'], multi_scale=True, cuda=config['cuda'])

    # Calculate the detection result for each image in hpatches dataset.
    # For each sequence in hpatches dataset.
    for seq_name in sorted(os.listdir(config['image_dir'])):
        seq_path = os.path.join(config['image_dir'], seq_name)
        for img_name in sorted(os.listdir(seq_path)):
            if img_name[-4:] != '.ppm':
                continue
            if config['cuda']:
                torch.cuda.empty_cache()
            img_path = os.path.join(seq_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0

            # I have modifiled the sekd output before.l
            keypoints, descriptors = feature_extractor.detectForEval(img)
            keypoints = keypoints
            scores = keypoints[2, :]
            keypoints = keypoints[0:2].T
            descriptors = descriptors.T

            det_dir = os.path.join(config['feature_dir'], seq_name)
            if not os.path.isdir(det_dir):
                os.makedirs(det_dir)
            if version == 0:
                det_path = os.path.join(det_dir, img_name + '.sekd_single')
            elif version == 1:
                det_path = os.path.join(det_dir, img_name + '.sekd_multi')
            print(det_path)
            with open(det_path, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints, scores=scores,
                    descriptors=descriptors)


RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]
norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])

def extract_r2d2(config={},version=0):
    print("Export R2D2 results:")
    outdir = config['feature_dir']
    # load the model
    net = r2d2.load_network('/home/yushichen/projects/r2d2/data/model_5e-5_9epoch.pt')
    net = net.cuda()

    for seq_name in sorted(os.listdir(config['image_dir'])):
        seq_path = os.path.join(config['image_dir'], seq_name)
        for img_name in sorted(os.listdir(seq_path)):
            if img_name[-4:] != '.ppm':
                continue
            if config['cuda']:
                torch.cuda.empty_cache()
            img_path = os.path.join(seq_path, img_name)
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img = img.astype(np.float32) / 255.0
            img = Image.open(img_path).convert('RGB')
            W,H = img.size
            img = norm_RGB(img)[None]
            img = img.cuda()
            print('img_shape',img.shape)
            if version == 0:
                keypoints, descriptors, scores = r2d2_val(img=img, net=net, version=0)
            elif version == 1:
                keypoints, descriptors, scores = r2d2_val(img=img, net=net, version=1)
            print('keypoint: ',keypoints.shape)

            det_dir = os.path.join(config['feature_dir'], seq_name)
            if not os.path.isdir(det_dir):
                os.makedirs(det_dir)
            if version == 0:
                det_path = os.path.join(det_dir, img_name + '.myr2d2_single')
            elif version == 1:
                det_path = os.path.join(det_dir, img_name + '.myr2d2_multi')
            print(det_path)
            with open(det_path, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints, scores=scores,
                    descriptors=descriptors)

