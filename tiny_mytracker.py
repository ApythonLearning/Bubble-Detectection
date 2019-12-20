import cv2 as cv

from mytracker import *
tracker = MyTracker()


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
    """
    output all the bounding box in a frame

    """
    dets = []
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dst = np.zeros(img.shape, dtype=np.uint8)
    if len(contours) > 200:
        print(len(contours))
        return dets
    for i in range(len(contours)):
        cv.drawContours(dst, contours, i, (255, 0, 0), 1)
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area <= 80:
            continue
        x, y, w, h = cv.boundingRect(cnt)

        if x < 2 and y < 2:
            continue
        cv.rectangle(video_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        dets.append([x, y, x + w, y + h, 0.8])
    # dets = np.asarray(dets)

    cv.imshow('contours', dst)
    cv.imshow('bbox', video_frame)

    return dets


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


def contrast_brightness_demo(img, ctr, bri):  # 亮度b 对比度c
    """
    Simply change the contrast and brightness of a image.
    But I am not sure if there is a better way.
    """
    blank = np.zeros(img.shape, img.dtype)
    out = cv.addWeighted(img, ctr, blank, 1 - ctr, bri)

    return out


Color = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
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
    # if not framecount:
    #     box = select_obj(frame, 'origin')  # ROI选取
    # frame = region_resize(frame, box, Size)
    origin_frame = frame
    smooth_frame = cv.medianBlur(frame, 3)
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

    ret, binary1 = cv.threshold(sub_frame1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    ret1, binary2 = cv.threshold(sub_frame2, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 对前景进行形态学操作
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel_e)
    return_value, fgmask = cv.threshold(fgmask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    binary = cv.bitwise_and(binary1, binary2)
    canny = cv.Canny(frame_augment, 100, 300)
    binary = cv.bitwise_or(binary, canny)
    binary = cv.bitwise_or(binary, fgmask)
    cv.imshow('binary', binary)

    # 轮廓绘制
    # binary = cv.dilate(binary, kernel_e)
    # morph_binary = cv.erode(binary, kernel_e)
    morph_binary = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel_e)
    cv.imshow('second', morph_binary)
    cnt_box = draw_obj(morph_binary, origin_frame)

    if not len(cnt_box):
        continue
    tracker.load_boxes(cnt_box)

    if framecount >= 4:
        tracks, collision_split_num = tracker.updates()
    else:
        Spre_frame = pre_frame
        pre_frame = frame
        continue

    for i, track in enumerate(tracks):
        (x, y) = (int(track[0]), int(track[1]))
        (w, h) = (int(track[2]), int(track[3]))
        (b, g, r) = Color[i, :]
        cv.rectangle(origin_frame, (x, y), (w, h), (int(b), int(g), int(r)), 2)
        cv.putText(origin_frame, str(i), (x, y - 5), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (int(b), int(g), int(r)), 2)


    # boxes = []
    # index_id = []
    # previous = memory.copy()
    # memory = {}
    #
    # for track in tracks:
    #     boxes.append([track[0], track[1], track[2], track[3]])
    #     index_id.append(int(track[4]))
    #     memory[index_id[-1]] = boxes[-1]
    # print(memory)
    #
    # for str_id, key in enumerate(memory):
    #     num_id = int(key)
    #     (x, y) = (int(memory[key][0]), int(memory[key][1]))
    #     (w, h) = (int(memory[key][2]), int(memory[key][3]))
    #     (b, g, r) = Color[num_id, :]
    #
    #     cv.rectangle(origin_frame, (x, y), (w, h), (int(b), int(g), int(r)), 2)
    #     cv.putText(origin_frame, str(key), (x, y-5), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (int(b), int(g), int(r)), 2)

    cv.imshow('sort', origin_frame)

    # 更新帧
    Spre_frame = pre_frame
    pre_frame = frame

    c = cv.waitKey(10)
    if c == 27:
        break

cv.waitKey(0)
capture.release()
cv.destroyAllWindows()
