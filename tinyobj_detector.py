import cv2 as cv
import numpy as np


def select_obj(img, window_name):
    # cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    r = cv.selectROI(window_name, img)

    return r


def region_resize(img, r, size):
    # print(img)
    imgCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    imgCrop = cv.resize(imgCrop, size)

    return imgCrop


def draw_obj(img, video_frame):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dst = np.zeros(img.shape, dtype=np.uint8)
    if len(contours) > 200:
        print(len(contours))
        return dst
    for i in range(len(contours)):
        cv.drawContours(dst, contours, i, (255, 0, 0), 1)
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area <= 20:
            continue
        x, y, w, h = cv.boundingRect(cnt)

        if x < 2 and y < 2:
            continue
        cv.rectangle(video_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('contours', dst)
    cv.imshow('bbox', video_frame)

    return dst


def correct_draw(image, video_frame, cor_box, size):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dst = np.zeros(image.shape, dtype=np.uint8)
    if len(contours) > 200:
        print(len(contours))
        return dst
    for i in range(len(contours)):
        cv.drawContours(dst, contours, i, (255, 0, 0), 1)
        cnt = contours[i]
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(video_frame, (x+cor_box[1], y+cor_box[0]),
                     (x + cor_box[1] + int(w*cor_box[3]/size[0]), y + cor_box[0] + int(h*cor_box[2]/size[1])),
                     (0, 0, 255), 2)

    cv.imshow('contours', dst)
    cv.imshow('bbox', video_frame)

    return dst


# 简单亮度对比度调整
def contrast_brightness_demo(img, ctr, bri):  # 亮度b 对比度c
    blank = np.zeros(img.shape, img.dtype)
    out = cv.addWeighted(img, ctr, blank, 1 - ctr, bri)

    return out


Size = (500, 700)  # resize size
fgbg = cv.createBackgroundSubtractorMOG2()
capture = cv.VideoCapture('E:/cvdata/resized_avi/new_test1.avi')
framecount = 0
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
kernel_e = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel_new = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
kernel_1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))

while True:
    ret, frame = capture.read()
    if not framecount:
        box = select_obj(frame, 'origin')  # ROI选取

    # frame = region_resize(frame, box, Size)
    origin_frame = frame
    smooth_frame = cv.medianBlur(frame, 3)  # 中值滤波在一定程度上会影响背景减法效果
    fgmask = fgbg.apply(smooth_frame)

    if ret is False:
        break

    cv.imshow('origin', frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (3, 3), 15)
    #     frame = cv.medianBlur(frame, 3)
    framecount += 1
    if framecount == 1:
        Spre_frame = frame
        continue
    if framecount == 2:
        pre_frame = frame
        continue

    sub_frame1 = cv.absdiff(Spre_frame, pre_frame)
    sub_frame2 = cv.absdiff(pre_frame, frame)

    frame_augment = contrast_brightness_demo(sub_frame2, 3, 3)

    sub_frame1 = cv.medianBlur(sub_frame1, 3)
    sub_frame2 = cv.medianBlur(sub_frame2, 3)

    ret, binary1 = cv.threshold(sub_frame1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    ret1, binary2 = cv.threshold(sub_frame2, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 对前景进行形态学操作
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel_new)
    return_value, fgmask = cv.threshold(fgmask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    binary = cv.bitwise_and(binary1, binary2)
    canny = cv.Canny(frame_augment, 100, 300)
    binary = cv.bitwise_or(binary, canny)
    binary = cv.bitwise_or(binary, fgmask)
    cv.imshow('canny', canny)
    cv.imshow('binary', binary)
    cv.imshow('fgmask', fgmask)

    # 轮廓绘制
    # binary = cv.dilate(binary, kernel_e)
    # morph_binary = cv.erode(binary, kernel_e)
    morph_binary = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel_e)  # 改变形态学操作的核大小可以有效消除背景减法带来的噪声
    cv.imshow('second', morph_binary)
    obj_contours = draw_obj(morph_binary, origin_frame)

    # 更新帧
    Spre_frame = pre_frame
    pre_frame = frame

    c = cv.waitKey(10)
    if c == 27:
        break

cv.waitKey(0)
capture.release()
cv.destroyAllWindows()
