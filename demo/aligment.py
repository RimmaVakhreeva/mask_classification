from typing import List

import cv2
import numpy as np
from skimage import transform as trans

ARCFACE_SRC = np.array(
    [[[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
      [41.5493, 92.3655], [70.7299, 92.2041]]],
    dtype=np.float32)


def estimate_norm(landmarks):
    assert landmarks.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(landmarks, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    for i in np.arange(ARCFACE_SRC.shape[0]):
        tform.estimate(landmarks, ARCFACE_SRC[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - ARCFACE_SRC[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def face_align(image: np.ndarray,
               landmarks: np.ndarray,
               image_size: int = 112) -> List[np.ndarray]:
    if len(landmarks) == 0:
        return []

    output = []
    for landmark in landmarks:
        M, pose_index = estimate_norm(landmark)
        warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
        output.append(warped)
    return output
