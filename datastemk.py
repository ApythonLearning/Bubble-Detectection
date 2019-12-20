import cv2 as cv
import numpy as np
import os
import random


# os.makedir("F:/study/vediodemo/imagesets")
# capture = cv.VideoCapture("F:/study/vediodemo/DZSX1.5-1.avi")
# height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
# width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
# count = capture.get(cv.CAP_PROP_FRAME_COUNT)
# fps = capture.get(cv.CAP_PROP_FPS)
# print(height, width, count, fps)
# framecount = 0
#
# while True:
#     ret, frame = capture.read()
#     if ret is True:
#         framecount = framecount + 1
#         cv.imshow('videodemo', frame)
#         if framecount % 10 == 0:
#             cv.imwrite('E:/cvdata/datasets/t1_video_0001_00'+str(int(framecount/10))+'.jpg', frame)
#         c = cv.waitKey(50)
#         if c == 27:
#             break
#     else:
#         break
# capture.release()
# cv.destroyAllWindows()
os.chdir("D:/obj_detect/yolov3/yolov3/")
name_list = os.listdir("D:/obj_detect/yolov3/yolov3/JPEGImages-1/")
f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')
num_list = [x for x in range(len(name_list))]
random.shuffle(num_list)
length = len(num_list)
for i in range(0, int(length*4/5)):
    f1.write('data/object/'+name_list[num_list[i]]+'\n')
f1.close()
for index in range(int(length * 4 / 5), len(num_list)):
    f2.write('data/object/'+name_list[num_list[index]]+'\n')
f2.close()


