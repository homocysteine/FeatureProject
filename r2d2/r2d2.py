from .r2d2_extraction import NonMaxSuppression,\
    extract_multiscale,load_network,extract_singlescale
import time

class R2D2_Creator:
    def __init__(self):
        super(R2D2_Creator, self).__init__()
        self.model = load_network('../model/r2d2_WASF_N16.pt')
        self.model = self.model.cuda()

    def detectAndCompute(self,img):
        detector = NonMaxSuppression(rel_thr=0.7,rep_thr=0.7)

        start_time = time.time()
        # xys, desc, scores = extract_multiscale(self.model, img, detector,
        #     scale_f=2 ** 0.25,
        #     min_scale=0,
        #     max_scale=1,
        #     min_size=256,
        #     max_size=1024,
        #  verbose=False)

        xys, desc, scores = extract_singlescale(self.model,img,detector)
        end_time = time.time()
        # print('extracting time',end_time-start_time)

        idxs = scores.argsort()[-500:]
        xys = xys[idxs]
        desc = desc[idxs]
        scores = scores[idxs]

        kp = xys[:, :2].cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()

        # print(type(kp), kp.shape, kp[0])
        # print(type(desc), desc.shape)
        # print(type(scores), scores.shape)
        # print(img.shape)

        return kp, desc, scores

def r2d2_val(img, net, version=0):
    print(img.shape)
    detector = NonMaxSuppression(rel_thr=0.7, rep_thr=0.7)
    # extract keypoints/descriptors for a single image
    if version == 1:
        xys, desc, scores = extract_multiscale(net, img, detector,
            scale_f=2**0.25,
            min_scale=0,
            max_scale=1,
            min_size=256,
            max_size=1024,
            verbose=False)
    elif version == 0:
        xys, desc, scores = extract_singlescale(net, img, detector)

    idxs = scores.argsort()[-500:]
    xys = xys[idxs]
    desc = desc[idxs]
    scores = scores[idxs]

    kp = xys[:,:2].cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()

    print(type(kp),kp.shape,kp[0])
    print(type(desc),desc.shape)
    print(type(scores),scores.shape)
    print(img.shape)

    return kp, desc, scores


