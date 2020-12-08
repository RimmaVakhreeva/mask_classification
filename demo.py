import cv2

from demo.retinaface import create_retinaface, retinaface_forward

image = cv2.imread('/Users/rimmavahreeva/Desktop/test.jpg')
model = create_retinaface('/Users/rimmavahreeva/PycharmProjects/mask_classification/'
                              'demo/checkpoints/mobilenet0.25_Final.pth', cpu=True)
bboxes, scores, landmarks = retinaface_forward(model, image, confidence_threshold=0.5)
