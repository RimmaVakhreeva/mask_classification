import cv2
import torch

import albumentations as A
import albumentations.pytorch as AP

from demo.aligment import face_align
from demo.retinaface import create_retinaface, retinaface_forward
from mask_classification.classifier.create_model import create_model
from mask_classification.classifier.module import ClassifierLightning

test_transfoms = A.Compose([
    A.LongestMaxSize(max_size=112),
    A.PadIfNeeded(min_width=112, min_height=112, value=(128, 128, 128), border_mode=0),
    A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.),
    AP.transforms.ToTensorV2()
])

def _conf_2_label(conf: float) -> str:
    if conf <= 0.3:
        return 'no_mask'
    elif 0.3 < conf < 0.5:
        return 'unknown'
    else:
        return 'with_mask'


def classifier_forward(model, images):
    input_images = [
        test_transfoms(image=img)['image']
        for img in images
    ]

    with torch.no_grad():
        output_classifications = model(torch.stack(input_images))
    output_classifications = output_classifications.cpu().numpy()

    import numpy as np
    print((output_classifications * 100).astype(np.int))

    return [
        _conf_2_label(cls)
        for cls in output_classifications
    ]


def _draw_bboxes(image, bboxes, scores, landmarks, labels):
    for (x1, y1, x2, y2), score, lmks, label in zip(bboxes, scores, landmarks, labels):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, label, (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

        cv2.circle(image, (lmks[0, 0], lmks[0, 1]), 4, (0, 0, 255), 1)
        cv2.circle(image, (lmks[1, 0], lmks[1, 1]), 4, (0, 255, 255), 1)
        cv2.circle(image, (lmks[2, 0], lmks[2, 1]), 4, (255, 0, 255), 1)
        cv2.circle(image, (lmks[3, 0], lmks[3, 1]), 4, (0, 255, 0), 1)
        cv2.circle(image, (lmks[4, 0], lmks[4, 1]), 4, (255, 0, 0), 1)


capture = cv2.VideoCapture(0)

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 5, (frame_width, frame_height))
retinaface_model = create_retinaface('/Users/rimmavahreeva/PycharmProjects/mask_classification/'
                          'demo/checkpoints/mobilenet0.25_Final.pth', cpu=True)

while True:
    is_read, image = capture.read()
    if not is_read:
        break

    bboxes, scores, landmarks = retinaface_forward(retinaface_model, image, confidence_threshold=0.9)
    if len(bboxes) == 0:
        continue

    landmarks = landmarks.reshape((-1, 5, 2))
    warped_images = face_align(image, landmarks, image_size=112)
    assert len(warped_images) > 0

    classifier_model = ClassifierLightning.load_from_checkpoint(
        '/Users/rimmavahreeva/Desktop/Face_mask_detection/epoch=49.ckpt',
        model=create_model(pretrained=False, backbone_type='mobilenetv3_large_100', num_classes=1))
    classifier_model.eval()

    labels = classifier_forward(classifier_model, warped_images)

    _draw_bboxes(image, bboxes, scores, landmarks, labels)

    writer.write(image)
    cv2.imshow('image', image)
    if cv2.waitKey(1) == 27:
        break

capture.release()
writer.release()
