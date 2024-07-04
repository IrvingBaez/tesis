import sys
import cv2, os
import numpy as np
from skimage import transform as trans
from retinaface.pre_trained_models import get_model


src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)
# lmk is prediction; src is template

class face_aligner():
    def __init__(self):
        super().__init__()
        self.detector = get_model("resnet50_2020-07-20", max_size=2048, device='cuda')

    def estimate_norm(self, lmk, image_size=224, mode='arcface'):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        if mode == 'arcface':
            assert image_size == 224
            src = arcface_src
        else:
            src = src_map[image_size]
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
            # print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index


    def norm_crop(self, img, landmark, image_size=224, mode='arcface'):
        M, pose_index = self.estimate_norm(landmark, image_size, mode)
        # print(pose_index)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    def align_face(self, img, thresh=0.8):
        img = cv2.resize(img, (224, 224))

        predictions = self.detector.predict_jsons(img, thresh)
        faces = np.array([prediction['bbox'] for prediction in predictions])
        landmarks = np.array([prediction['landmarks'] for prediction in predictions])


        faces = faces.mean(axis=0)[None, :]
        landmarks = landmarks.mean(axis=0)[None, :]

        if faces is not None:
            try:
                img = self.norm_crop(img, landmarks[0], mode='custom')
            except:
                img = None
        else:
            img = None
        return img

    def cleanup(self):
        # Explicitly release GPU resources if necessary
        if hasattr(self.detector, 'model'):
            del self.detector.model
        if hasattr(self.detector, 'device'):
            self.detector.device = None


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout