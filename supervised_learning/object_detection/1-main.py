#!/usr/bin/env python3

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import cv2 as cv
    import numpy as np
    Yolo = __import__('1-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]], # large scale
                        [[30, 61], [62, 45], [59, 119]], # med scale
                        [[10, 13], [16, 30], [33, 23]]]) # small scale
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    output1 = np.random.randn(13, 13, 3, 85) # large
    output2 = np.random.randn(26, 26, 3, 85) # med
    output3 = np.random.randn(52, 52, 3, 85) # small

    # image = cv.imread("000000000001.jpg")
    # image = mpimg.imread("000000000001.jpg")
    # print("image shape", image.shape)
    # plt.imshow(image)
    # plt.show()
    # print("This is the type of yolo", type(yolo.model))
    # result = yolo.model.predict(image)
    # print("result shape", result.shape)
    # print("this is result", result)
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    # boxes = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    print("this is the shape of the boxe 1", boxes[0].shape)
    print("this is the shape of the boxe 2", boxes[1].shape)
    print('Boxes:', boxes)
    print('Box confidences:', box_confidences)
    print('Box class probabilities:', box_class_probs)