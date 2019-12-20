import cv2 as cv
import numpy as np


Size = (600, 800)
capture = cv.VideoCapture('F:/study/vediodemo/XS4.2-7.avi')
out = cv.VideoWriter('E:/cvdata/resized_avi/re_xs4.2-7.avi', cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 10,
                     (np.int(Size[0]), np.int(Size[1])), True)  # 此处size要与frame一致
frame_count = 0

while True:
    ret, frame = capture.read()
    if not ret:
        break
    if not frame_count:
        r = cv.selectROI('origin', frame)

    frame_count += 1

    cv.imshow('origin', frame)

    frame_crop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    frame_crop = cv.resize(frame_crop, Size)

    out.write(frame_crop)
    cv.imshow('crop', frame_crop)

    c = cv.waitKey(10)
    if c == 27:
        break

capture.release()
out.release()
cv.destroyAllWindows()


