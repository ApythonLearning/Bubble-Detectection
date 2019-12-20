import cv2 as cv
from matplotlib import pyplot as plt
import mytracker as myt
from Sort import *
tracker = Sort()
memory = {}
My_counter = myt.MyTracker()


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
    if len(contours) > 20000:
        print(len(contours))
        return dets
    for i in range(len(contours)):
        cv.drawContours(dst, contours, i, (255, 0, 0), 1)
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area <= 60:
            continue
        x, y, w, h = cv.boundingRect(cnt)

        if x < 1 and y < 1:
            continue
        cv.rectangle(video_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        dets.append([x, y, x + w, y + h, 0.9])
    dets = np.asarray(dets)

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


def iou_suppression(cnt_box, yolo_box, max_threshold, min_threshold):
    """
    To calculate the iou scores between cnt boxes and YoLo boxes, if the score surpass the
    appointed threshold we suppression the box of cnt boxes.
    :param cnt_box: np array
    :param yolo_box: np array
    :param max_threshold: threshold value
    :return: the all predicted boxes of a frame
    """
    all_boxes = []
    pre_bboxes = yolo_box
    bboxes = cnt_box
    for i in range(len(pre_bboxes)):
        max_flag = 0
        min_flag = 0
        for j in range(len(bboxes)):

            (pre_x1, pre_y1) = (pre_bboxes[i][0], pre_bboxes[i][1])
            (pre_x2, pre_y2) = (pre_bboxes[i][2], pre_bboxes[i][3])
            (cur_x1, cur_y1) = (bboxes[j][0], bboxes[j][1])
            (cur_x2, cur_y2) = (bboxes[j][2], bboxes[j][3])
            origin_w = pre_x2 - pre_x1
            origin_h = pre_y2 - pre_y1
            current_w = cur_x2 - cur_x1
            current_h = cur_y2 - cur_y1
            prime_area = origin_h * origin_w
            current_area = current_h*current_w

            if pre_x1 > cur_x1:
                if pre_y1 > cur_y1:
                    if cur_x2 - pre_x1 <= 0 or cur_y2 - pre_y1 <= 0:
                        lap_area = 0
                    else:
                        width = cur_x2 - pre_x1
                        height = cur_y2 - pre_y1
                        if width > origin_w:
                            width = origin_w
                        if height > origin_h:
                            height = origin_h

                        lap_area = width*height

                else:
                    if cur_x2 - pre_x1 <= 0 or pre_y2 - cur_y1 <= 0:
                        lap_area = 0
                    else:
                        width = cur_x2 - pre_x1
                        height = pre_y2 - cur_y1
                        if width > origin_w:
                            width = origin_w
                        if height > current_h:
                            height = current_h

                        lap_area = width*height
            else:
                if pre_y1 > cur_y1:
                    if pre_x2 - cur_x1 <= 0 or cur_y2 - pre_y1 <= 0:
                        lap_area = 0
                    else:
                        width = pre_x2 - cur_x1
                        height = cur_y2 - pre_y1
                        if width > current_w:
                            width = current_w
                        if height > origin_h:
                            height = origin_h

                        lap_area = width*height
                else:
                    if pre_x2 - cur_x1 <= 0 or pre_y2 - cur_y1 <= 0:
                        lap_area = 0
                    else:
                        width = pre_x2 - cur_x1
                        height = pre_y2 - cur_y1
                        if width > current_w:
                            width = current_w
                        if height > current_h:
                            height = current_h

                        lap_area = width*height

            if lap_area != 0:
                sum_area = (prime_area + current_area - lap_area)
                iou_score = lap_area/sum_area
                if iou_score > max_threshold:  # set the threshold of the iou scores, in line with the sort
                    max_flag = 1
                elif iou_score > min_threshold:
                    min_flag = 1

        if max_flag == 1 or min_flag == 0:
            all_boxes.append(pre_bboxes[i])

        if cnt_box != []:
            for index_box in range(cnt_box.shape[0]):
                all_boxes.append(cnt_box[index_box])

    return np.asarray(all_boxes)


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)

    return o


def iou_suppression_new(cnt_box_input, yolo_box, threshold_value):

    final_box_outs = []

    for cnt in range(cnt_box_input.shape[0]):
        flag_none = 0
        for cov in range(yolo_box.shape[0]):
            iou_cur = iou(cnt_box_input[cnt], yolo_box[cov])
            if iou_cur >= threshold_value:
                flag_none = 1
                break
        if flag_none == 0:
            final_box_outs.append(cnt_box_input[cnt])

    final_box_outs = np.vstack((np.asarray(final_box_outs), yolo_box))

    return final_box_outs


def ratio_area_calculate(inpt_bboxes):
    """
    calculate the aspect ration and area
    :param inpt_bboxes:
    :return:
    """
    bbox_width = inpt_bboxes[2] - inpt_bboxes[0]
    bbox_height = inpt_bboxes[3] - inpt_bboxes[1]
    aspect_ratio = bbox_width / bbox_height
    bbox_area = bbox_height * bbox_width

    return bbox_area, aspect_ratio


def cluster_classify(bounding_boxes):
    """
    Using the cluster algorithm to classify the size of the bubble, by using the
    area and aspect ratio of the bounding boxes as our features.
    :param bounding_boxes: bounding box coordinates of left-top point and right-bottom point.
    :return:the classification result
    """
    features = []
    for box_index in range(bounding_boxes.shape[0]):
        area, ratio = ratio_area_calculate(bounding_boxes[box_index])
        features.append([area, ratio])

    features_a = np.asarray(features, dtype=np.float32)
    print(features_a.shape)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(features_a, 4, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center_list = []
    for ctr in range(center.shape[0]):
        center_list.append(center[ctr, 0])
    sort_list = center_list.copy()
    sort_list.sort()
    index_list = []
    for element in sort_list:
        index_list.append(center_list.index(element))

    small = features_a[label.ravel() == index_list[0]]
    middle = features_a[label.ravel() == index_list[1]]
    large = features_a[label.ravel() == index_list[2]]
    poly = features_a[label.ravel() == index_list[3]]

    plt.scatter(small[:, 0], small[:, 1])
    plt.scatter(middle[:, 0], middle[:, 1], c='r', marker='s')
    plt.scatter(large[:, 0], large[:, 1], c='g', marker='d')
    plt.scatter(poly[:, 0], poly[:, 1], c='b', marker='^')
    plt.legend(['small bubble', 'middle bubble', 'large bubble', 'poly bubble'])
    plt.xlabel('area')
    plt.ylabel('aspect ratio')

    plt.show()

    return


yolo_tiny_model = "D:/obj_detect/darknet/build/darknet/x64/backup/yolov3-tiny-obj-my_4000.weights"
yolo_tiny_cfg = "D:/obj_detect/darknet/build/darknet/x64/cfg/yolov3-tiny-obj-my.cfg"

classes = None
with open("D:/obj_detect/darknet/build/darknet/x64/data/yolov3-tiny-obj-names.txt", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv.dnn.readNetFromDarknet(yolo_tiny_cfg, yolo_tiny_model)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

Color = np.random.randint(0, 255, (10000, 3), dtype=np.uint8)
fgbg = cv.createBackgroundSubtractorMOG2()

capture = cv.VideoCapture('E:/cvdata/resized_avi/new_test2.avi')
frame_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
frame_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
framecount = 0

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
kernel_e = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel_new = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
kernel_1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))

while True:
    ret, frame = capture.read()
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

    # 轮廓绘制
    contour_frame = origin_frame.copy()
    ground_truth = origin_frame.copy()
    combine_frame = origin_frame.copy()
    morph_binary = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel_e)
    cnt_box = draw_obj(morph_binary, contour_frame)
    contour_transient = contour_frame.copy()

    h, w = origin_frame.shape[:2]
    # 基于多个Region层输出getUnconnectedOutLayersNames
    blobImage = cv.dnn.blobFromImage(origin_frame, 1.0 / 255.0, (416, 416), None, True, False)
    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    outs = net.forward(outNames)

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
            if confidence > 0.2:
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
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)  # 0.5 confidence threshold 0.4 NMS scores threshold
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv.rectangle(origin_frame, (left, top), (left + width, top + height), (0, 255, 0), 2, 8, 0)
        dets_yolo.append([left, top, left + width, top + height, confidences[i]])
    dets_yolo = np.asarray(dets_yolo)
    yolo_transient = origin_frame.copy()

    # out_put_boxes = iou_suppression(cnt_box, dets_yolo, 0.8, 0.1)
    combine_out = iou_suppression_new(cnt_box, dets_yolo, 0.1)
    print('combine is:\n ', combine_out)
    for i in range(combine_out.shape[0]):
        cv.rectangle(combine_frame, (int(combine_out[i][0]), int(combine_out[i][1])),
                     (int(combine_out[i][2]), int(combine_out[i][3])), (0, 255, 255), 2)
    cv.imshow('combine', combine_frame)

    final_boxes = combine_out
    #  tracking part
    if not len(final_boxes):
        continue
    tracks = tracker.update(final_boxes)

    boxes_tracker = []
    index_id = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes_tracker.append([track[0], track[1], track[2], track[3]])
        index_id.append(int(track[4]))
        memory[index_id[-1]] = boxes_tracker[-1]
    print(memory)

    for str_id, key in enumerate(memory):
        num_id = int(key)
        (x, y) = (int(memory[key][0]), int(memory[key][1]))
        (w, h) = (int(memory[key][2]), int(memory[key][3]))
        (b, g, r) = Color[num_id, :]

        cv.rectangle(origin_frame, (x, y), (w, h), (int(b), int(g), int(r)), 2)
        cv.putText(origin_frame, str(key), (x, y-5), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (int(b), int(g), int(r)), 2)

    cv.imshow('sort', origin_frame)

    my_cnt_box = cnt_box.tolist()
    if not len(my_cnt_box):
        continue
    My_counter.load_boxes(my_cnt_box)

    if framecount >= 4:
        tracks_my_tracker, collision_split_num = My_counter.updates()
    else:
        Spre_frame = pre_frame
        pre_frame = frame
        continue

    # 更新帧
    Spre_frame = pre_frame
    pre_frame = frame

    c = cv.waitKey(10)
    if c == 27:
        break

    if c == 32:
        cv.imwrite('E:/cvdata/transient/cnt_solo.png', contour_transient)
        cv.imwrite('E:/cvdata/transient/yolo_solo.png', yolo_transient)
        cv.imwrite('E:/cvdata/transient/ground_truth.png', ground_truth)
        cv.imwrite('E:/cvdata/transient/combine_result.png', combine_frame)
        cluster_classify(cnt_box)

cv.waitKey(0)
capture.release()
cv.destroyAllWindows()
