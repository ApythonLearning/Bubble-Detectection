import cv2 as cv
import numpy as np
import os
from cnn import *

os.chdir('D:/obj_detect/darknet/build/darknet/x64/')
yolo_tiny_model = "backup/yolov3-tiny-obj-my_4000.weights"
yolo_tiny_cfg = "cfg/yolov3-tiny-obj-my.cfg"

classes = None
with open("D:/obj_detect/darknet/build/darknet/x64/data/yolov3-tiny-obj-names.txt", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# print(classes)
Color = [(0, 0, 255), (255, 255, 0)]

net = cv.dnn.readNetFromDarknet(yolo_tiny_cfg, yolo_tiny_model)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

cap = cv.VideoCapture("E:/cvdata/resized_avi/new_test2.avi")
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
index = 0
while True:
    ret, image = cap.read()
    if ret is False:
        break
    #     image = cv.flip(image, 1)
    h, w = image.shape[:2]
    origin = image.copy()
    # 基于多个Region层输出getUnconnectedOutLayersNames
    blobImage = cv.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), None, True, False)
    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    outs = net.forward(outNames)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    fps = 1000 / (t * 1000.0 / cv.getTickFrequency())
    label = 'FPS: %.2f' % fps
    # cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 绘制检测矩形
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # numbers are [center_x, center_y, width, height]
            if confidence > 0.1:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # 使用非最大抑制
    dets_yolo = []
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)  # 0.5 confidence threshold 0.4 NMS scores threshold
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv.rectangle(image, (left, top), (left + width, top + height), Color[classIds[i]], 2, 8, 0)
        # cv.putText(image, classes[classIds[i]], (left, top),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, Color[classIds[i]], 2)
        dets_yolo.append([left, top, left + width, top + height, confidences[i]])
    dets_yolo = np.asarray(dets_yolo)

    c = cv.waitKey(1)
    if c == 27:
        break
    index += 1
    cv.imshow('YOLOv3-tiny-Detection-Demo', image)

    if c == 32 or index == 881:
        # cluster_classify(dets_yolo)
        cv.imwrite("E:/cvdata/paper_demo/tiny.jpg", image)
        cv.imwrite("E:/cvdata/paper_demo/tiny-origin.jpg", origin)

cv.waitKey(0)
cv.destroyAllWindows()
