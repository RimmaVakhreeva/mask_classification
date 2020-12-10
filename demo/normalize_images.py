from logging import warning
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import cv2

from demo.aligment import face_align
from demo.retinaface import create_retinaface, retinaface_forward

images_with_mask = Path('/Users/rimmavahreeva/Desktop/Face_mask_detection/mafa/crops/mask')
#images_without_mask = Path('/Users/rimmavahreeva/Desktop/Face_mask_detection/test/without_mask')

normalize_images_with_mask = Path('/Users/rimmavahreeva/Desktop/Face_mask_detection/normalize_mafa')
#normalize_images_without_mask = Path('/Users/rimmavahreeva/Desktop/Face_mask_detection/test_normalize_images/without_mask')

img_suffixes = {'.png', '.jpg', '.jpeg'}
for image_filename in tqdm(images_with_mask.iterdir()):
    if image_filename.suffix.lower() not in img_suffixes:
        continue
    image = cv2.imread(str(image_filename))
    model = create_retinaface('/Users/rimmavahreeva/PycharmProjects/mask_classification/'
                              'demo/checkpoints/mobilenet0.25_Final.pth', cpu=True)
    bboxes, scores, landmarks = retinaface_forward(model, image[..., ::-1], confidence_threshold=0.5)
    if len(bboxes) == 0:
        warning(f'Not found faces on {image_filename}')
        continue

    landmarks = landmarks.reshape((-1, 5, 2))
    warped_images = face_align(image, landmarks, image_size=112)
    assert len(warped_images) > 0
    cv2.imwrite(str(normalize_images_with_mask / image_filename.name),
                warped_images[0])

