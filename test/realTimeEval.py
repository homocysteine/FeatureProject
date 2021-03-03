
import cv2
import matplotlib.cm as cm
import torch
import time

from utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor,frame2tensorRGB)
from feature_matching import Sift,Matching
import os

torch.set_grad_enabled(False)
def visualize(config,input=0,resize=[640,480],show_keypoints=False,no_display=False,force_cpu=False):
    if len(resize) == 2 and resize[1] == -1:
        resize = resize[0:1]
    if len(resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            resize[0], resize[1]))
    elif len(resize) == 1 and resize[0] > 0:
        print('Will resize max dimension to {}'.format(resize[0]))
    elif len(resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')


    vs1 = VideoStreamer(input, resize, 1,
                       image_glob=['*.png', '*.jpg', '*.jpeg'],i=0)
    vs2 = VideoStreamer(input, resize, 1,
                        image_glob=['*.png','*.jpg','*.jpeg'],i=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # torch.backends.cudnn.benchmark = True
    torch_set_gpu(0)
    matching = Matching(config=config).eval().to(device)

    # Create a window to display the demo.
    if not no_display:
        cv2.namedWindow('Visualization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Visualization', (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    totalTime = 0
    count = 0
    while True:
        frame1, ret1 = vs1.next_frame()
        frame2, ret2 = vs2.next_frame()
        count = count + 1

        if config.get('r2d2') != None:
            frame1_tensor = torch.from_numpy(frame1).float()[None].permute(0, 3, 1, 2).to('cuda')
            frame2_tensor = torch.from_numpy(frame2).float()[None].permute(0, 3, 1, 2).to('cuda')
        if config.get('superpoint') != None or config.get('sekd') !=None:
            # RGB image to gray image
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)
            frame1_tensor = frame2tensor(frame1,device)
            frame2_tensor = frame2tensor(frame2,device)
            if config.get('sekd') != None:
                frame1_tensor = frame1_tensor.cpu().numpy().squeeze()
                frame2_tensor = frame2_tensor.cpu().numpy().squeeze()
        if ret1 == False or ret2 == False:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = vs1.i, vs2.i
        # print('Image Index',stem0,stem1)


        start_time = time.time()
        if (config.get('r2d2') != None) or (config.get('superpoint') != None) or (config.get('sekd')) != None :
            kpts0, kpts1, matches, confidence = matching(frame1_tensor, frame2_tensor)
        else:
            kpts0, kpts1, matches, confidence = matching(frame1,frame2)
        end_time = time.time()
        timer.update('forward')

        valid = matches > -1
        #print(valid)
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        #print(confidence)
        color = cm.jet(confidence[valid])

        # print(mkpts0)
        timePerImage = end_time - start_time
        name ='sift_flann'
        text = [
            name,
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
            'FPS:{:.2f}'.format(1/(timePerImage)),
        ]
        k_thresh = 0.0
        m_thresh = 0.0
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        print('Length of mkpts',len(mkpts0))
        out = make_matching_plot_fast(
            frame1, frame2, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=show_keypoints, small_text=small_text)

        totalTime = totalTime + timePerImage
        print('final FPS:', count / totalTime)
        if not no_display:
            print(type(out),out.shape)
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs1.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'k':
                show_keypoints = not show_keypoints

        timer.update('viz')
        timer.print()

    cv2.destroyAllWindows()
    vs1.cleanup()


def convertFrame(isRGB, frame, device):
    if isRGB != None:
        frame_tensor = frame2tensorRGB(frame, device)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_tensor = frame2tensor(frame, device)  # to tensor;normalize
    return frame_tensor

def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu>=0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'],os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True # speed-up cudnn
        torch.backends.cudnn.fastest = True # even more speed-up?
        print( 'Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'] )
    else:
        print( 'Launching on CPU' )
    return cuda

if __name__ == '__main__':
    config = [
        {
            'sift': {},
            'flann': {}
        },
        {
            'orb': {},
            'flann': {}
        },
        {
            'r2d2':{},
            'brute-force':{}
        },
        {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 500,
            },
            'brute-force':{}
        },
        {
            'sekd':{},
            'flann':{}
        }
    ]
    # visualize(config=config[2],input='/media/yushichen/DATA/Datasets/motion.mp4')
    visualize(config=config[4], input='/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/01/image_0')
    # superglue='indoor',max_keypoints=240,keypoint_threshold=0.005,nms_radius=4,
    #               sinkhorn_iterations=20,match_threshold=0.2
