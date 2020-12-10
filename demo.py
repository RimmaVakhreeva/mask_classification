from logging import warning

import cv2

from demo.aligment import face_align
from demo.retinaface import create_retinaface, retinaface_forward


def _draw_bboxes(image, bboxes, scores, landmarks):
    for (x1, y1, x2, y2), score, lmks in zip(bboxes, scores, landmarks):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # landms
        cv2.circle(image, (lmks[0], lmks[1]), 4, (0, 0, 255), 1)
        cv2.circle(image, (lmks[2], lmks[3]), 4, (0, 255, 255), 1)
        cv2.circle(image, (lmks[4], lmks[5]), 4, (255, 0, 255), 1)
        cv2.circle(image, (lmks[6], lmks[7]), 4, (0, 255, 0), 1)
        cv2.circle(image, (lmks[8], lmks[9]), 4, (255, 0, 0), 1)


capture = cv2.VideoCapture(0)

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame_width, frame_height))
model = create_retinaface('/Users/rimmavahreeva/PycharmProjects/mask_classification/'
                              'demo/checkpoints/mobilenet0.25_Final.pth', cpu=True)

while True:
    is_read, image = capture.read()
    if not is_read:
        break

    bboxes, scores, landmarks = retinaface_forward(model, image, confidence_threshold=0.5)
    if len(bboxes) == 0:
        continue

    _draw_bboxes(image, bboxes, scores, landmarks)

    landmarks = landmarks.reshape((-1, 5, 2))
    warped_images = face_align(image, landmarks, image_size=112)
    assert len(warped_images) > 0



    writer.write(image)
    cv2.imshow('image', image)
    if cv2.waitKey(1) == 27:
        break

capture.release()
writer.release()
