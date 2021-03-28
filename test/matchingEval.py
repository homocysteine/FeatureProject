from utils import(compute_pose_error, compute_epipolar_error,
                          estimate_pose, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, frame2tensor, frame2tensorRGB)
from feature_matching import Matching
from pathlib import Path
import numpy as np
import cv2

def evaluate_matching_quality(resize=[800],input_pairs='',input_dir='',output_dir=''):
    if len(resize) == 2 and resize[1] == -1:
        resize = resize[0:1]
    if len(resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            resize[0], resize[1]))
    elif len(resize) == 1 and resize[0] > 0:
        print('Will resize max dimension to {}'.format(resize[0]))
    elif len(resize) == 1:
        print('Will not resize images!')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # read ground truth
    with open(input_pairs,'r') as f:
        pairs = [l.split() for l in f.readlines()]

    # eval
    if not all([len(p) == 38 for p in pairs]):
        raise ValueError(
            'All pairs should have ground truth info for evaluation.'
            'File \"{}\" needs 38 valid entries per row'.format(input_pairs))

    device = 'cuda'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'r2d2':{},
        'brute-force':{}
    }

    matching = Matching(config).eval().to(device)
    input_dir = Path(input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True,parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    print('Will write evaluation results',
          'to directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)
    print('Length of pairs',len(pairs))
    for i,pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0,stem1)

        do_match = True
        do_eval = True

        print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
        # print('pair',pair)
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0
        # print('rotation',rot0,rot1)
        # Load the image pair.
        resize_float = False
        image0, scales0 = read_image(
            input_dir / name0, device, resize, rot0, resize_float)
        image1, scales1 = read_image(
            input_dir / name1, device, resize, rot1, resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir / name0, input_dir / name1))
            exit(1)
        timer.update('load_image')

        if config.get('r2d2') != None:
            inp0 = frame2tensorRGB(image0,device)
            inp1 = frame2tensorRGB(image1,device)
        if config.get('superpoint') != None or config.get('sekd') != None:
            # RGB-> grayscale
            image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            inp0 = frame2tensor(image0, device)
            inp1 = frame2tensor(image1, device)
            if config.get('sekd') != None:
                inp0 = inp0.cpu().numpy().squeeze()
                inp1 = inp1.cpu().numpy().squeeze()

        if do_match:
            # print('image shape',inp0.shape)
            if (config.get('r2d2') != None) or \
                    (config.get('superpoint') != None) or \
                    (config.get('sekd') != None):
                kpts0, kpts1, matches, confidence = matching(inp0,inp1)
            else:
                kpts0, kpts1, matches, confidence = matching(image0, image1)
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]

        if do_eval:
            K0 = np.array(pair[4:13]).astype(float).reshape(3,3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3,3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4,4)

            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # update intrinsic matrix if rot value is not 0
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0


            # calculate error
            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            # f error is below 5e-4, we regard it as a correct matching
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) >0 else 0

            thresh = 1.
            # calculate pose(R t) with Essential
            print('Length of mkpts',len(mkpts0),name0)
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            # print('ret',ret)
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t , inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

    # calculate auc
    pose_errors = []
    precisions = []
    matching_scores = []
    for pair in pairs:
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        results = np.load(eval_path)
        pose_error = np.maximum(results['error_t'],results['error_R'])
        pose_errors.append(pose_error)
        precisions.append(results['precision'])
        matching_scores.append(results['matching_score'])

    # set different threshold
    thresholds = [5, 10, 20]
    print(pose_errors)
    aucs = pose_auc(pose_errors,thresholds)
    aucs = [100.* yy for yy in aucs]
    prec = 100.* np.mean(precisions)
    ms = 100.* np.mean(matching_scores)

    print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))

if __name__ == '__main__':
    evaluate_matching_quality(input_pairs='/home/yushichen/PycharmProjects/pythonProject/data/matching_eval/yfcc_test_pairs_with_gt_edition.txt',
                              input_dir='/media/yushichen/DATA/Datasets/OANet/raw_data/yfcc100m/',
                              output_dir='/home/yushichen/PycharmProjects/pythonProject/data/matching_eval/dump_yfcc_test_results/')














