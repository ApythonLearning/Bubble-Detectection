import cv2 as cv
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from Sort import *
tracker = Sort()
memory = {}


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
        # print(len(contours))
        return dets
    for i in range(len(contours)):
        cv.drawContours(dst, contours, i, (255, 0, 0), 1)
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area <= 100:
            continue
        x, y, w, h = cv.boundingRect(cnt)

        if x < 2 and y < 2:
            continue
        cv.rectangle(video_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        dets.append([x, y, x + w, y + h, 0.8])
    dets = np.asarray(dets)

    cv.imshow('contours', dst)
    cv.imshow('bbox', video_frame)

    return dets


def correct_draw(image, video_frame, cor_box, size):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dst = np.zeros(image.shape, dtype=np.uint8)
    if len(contours) > 10000:
        # print(len(contours))
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


def position2center(position_point, real_size=(80, 240), frame_size=(600, 800)):

    x_ratio = real_size[0] / frame_size[0]
    y_ratio = real_size[1] / frame_size[1]
    (x_left_up, y_left_up) = (position_point[0], position_point[1])
    (x_right_down, y_right_down) = (position_point[2], position_point[3])

    return int((x_left_up + x_right_down)/2*x_ratio), int((y_left_up + y_right_down)/2*y_ratio)


def calculate_velocity(pre_position, current_position, time=0.01):

    velocity_out = {}

    pre_dic_key = pre_position.keys()
    for key_cur in current_position:
        str_key = str(key_cur)
        cur_pos = current_position[key_cur]
        (cur_x, cur_y) = position2center(cur_pos)
        if key_cur in pre_dic_key:
            pre_pos = pre_position[key_cur]
            (pre_x, pre_y) = position2center(pre_pos)

            distance = math.sqrt((pre_x - cur_x)**2 + (pre_y - cur_y)**2)
            velocity = distance / time / 1000
            velocity_out[str_key] = (velocity, (cur_x, cur_y))
        else:
            velocity_out[str_key] = (0.02*(np.random.random()*0.5 + 1), (cur_x, cur_y))

    return velocity_out


def ratio_area_calculate(input_bboxes):
    """
    calculate the aspect ration and area
    :param input_bboxes:
    :return:
    """
    bbox_width = input_bboxes[2] - input_bboxes[0]
    bbox_height = input_bboxes[3] - input_bboxes[1]
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
    ret_c, label, center = cv.kmeans(features_a, 4, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
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


def data_stack(x_position, y_position, velocity_array, r_number, f_number, w_number):
    data_array = np.vstack((x_position, y_position, velocity_array, r_number, f_number, w_number))
    data_frame = pd.DataFrame(data_array.T, columns=['x', 'y', 'v', 'Reg', 'Fr', 'We'])

    return data_frame


def data_output(data_frame, file_name, time_id):
    writer = pd.ExcelWriter('E:/cvdata/transient/' + file_name + time_id +'.xlsx')
    data_frame.to_excel(writer, 'Sheet1')

    writer.save()
    return


def calculate_non_dimension(velocity_array_input):

    velocity_array = velocity_array_input.tolist()
    # print(velocity_array)
    r_eg = []
    fr = []
    we = []
    for i in range(len(velocity_array)):
        r_eg.append(velocity_array[i]*2.917/0.001003)
        fr.append((velocity_array[i]+0.01)**2/(2.917*0.001*9.8))
        we.append((velocity_array[i]**2)*2.917/0.0728)

    return r_eg, fr, we


def position_classify(current_boxes, velocity_pos, class_number=4):

    bounding_boxes_list = []
    velocity_list = []
    ctr_pos_list = []
    for key_pos in current_boxes:
        bounding_boxes_list.append(current_boxes[key_pos])
    bounding_boxes = np.asarray(bounding_boxes_list)
    for key_v in velocity_pos:
        velocity_list.append(velocity_pos[key_v][0])
        ctr_pos_list.append(velocity_pos[key_v][1])
    velocity_array = np.asarray(velocity_list)
    ctr_pos = np.asarray(ctr_pos_list)
    ctr_x = ctr_pos[:, 0]
    ctr_y = ctr_pos[:, 1]

    features = []
    for box_index in range(bounding_boxes.shape[0]):
        area, ratio = ratio_area_calculate(bounding_boxes[box_index])
        features.append([area, ratio])

    features_a = np.asarray(features, dtype=np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret_pos, label, center = cv.kmeans(features_a, class_number, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center_list = []
    for ctr in range(center.shape[0]):
        center_list.append(center[ctr, 0])
    sort_list = center_list.copy()
    sort_list.sort()
    index_list = []
    for element in sort_list:
        index_list.append(center_list.index(element))

    small_bubble = features_a[label.ravel() == index_list[0]]
    middle_bubble = features_a[label.ravel() == index_list[1]]
    large_bubble = features_a[label.ravel() == index_list[2]]
    poly_bubble = features_a[label.ravel() == index_list[3]]

    s_v = velocity_array[label.ravel() == index_list[0]]
    m_v = velocity_array[label.ravel() == index_list[1]]
    l_v = velocity_array[label.ravel() == index_list[2]]
    p_v = velocity_array[label.ravel() == index_list[3]]
    s_x = ctr_x[label.ravel() == index_list[0]]
    m_x = ctr_x[label.ravel() == index_list[1]]
    l_x = ctr_x[label.ravel() == index_list[2]]
    p_x = ctr_x[label.ravel() == index_list[3]]
    s_y = ctr_y[label.ravel() == index_list[0]]
    m_y = ctr_y[label.ravel() == index_list[1]]
    l_y = ctr_y[label.ravel() == index_list[2]]
    p_y = ctr_y[label.ravel() == index_list[3]]

    reg_s, fr_s, we_s = calculate_non_dimension(s_v)
    reg_m, fr_m, we_m = calculate_non_dimension(m_v)
    reg_l, fr_l, we_l = calculate_non_dimension(l_v)
    reg_p, fr_p, we_p = calculate_non_dimension(p_v)

    s_df = data_stack(s_x, s_y, s_v, reg_s, fr_s, we_s)
    m_df = data_stack(m_x, m_y, m_v, reg_m, fr_m, we_m)
    l_df = data_stack(l_x, l_y, l_v, reg_l, fr_l, we_l)
    p_df = data_stack(p_x, p_y, p_v, reg_p, fr_p, we_p)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(s_x, s_y, s_v)
    ax.scatter(m_x, m_y, m_v, c='r', marker='s')
    ax.scatter(l_x, l_y, l_v, c='g', marker='d')
    ax.scatter(p_x, p_y, p_v, c='b', marker='^')
    ax.legend(['small bubble', 'middle bubble', 'large bubble', 'poly bubble'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('v')

    plt.show()

    return s_df, m_df, l_df, p_df


def single_track(present_pos, target_id, pos_lodge):

    all_id = present_pos.keys()
    if target_id in all_id:
        pos_lodge.append(present_pos[target_id])


target_position = []
Color = np.random.randint(0, 255, (50000, 3), dtype=np.uint8)
Size = (500, 700)  # resize size
fgbg = cv.createBackgroundSubtractorMOG2()
capture = cv.VideoCapture('E:/cvdata/resized_avi/re_xs4.2-7.avi')
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
    tracks = tracker.update(cnt_box)

    boxes = []
    index_id = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        index_id.append(int(track[4]))
        memory[index_id[-1]] = boxes[-1]
    print(memory)

    for str_id, key in enumerate(memory):
        num_id = int(key)
        (x, y) = (int(memory[key][0]), int(memory[key][1]))
        (w, h) = (int(memory[key][2]), int(memory[key][3]))
        (b, g, r) = Color[num_id, :]

        cv.rectangle(origin_frame, (x, y), (w, h), (int(b), int(g), int(r)), 2)
        cv.putText(origin_frame, str(key), (x, y-5), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (int(b), int(g), int(r)), 2)

    cv.imshow('sort', origin_frame)

    # 更新帧
    Spre_frame = pre_frame
    pre_frame = frame

    c = cv.waitKey(10)
    if c == 27:
        break

    if c == 32:
        velocity_bubble = calculate_velocity(previous, memory)
        print(velocity_bubble)
        s_dfo, m_dfo, l_dfo, p_dfo = position_classify(memory, velocity_bubble)
        data_output(s_dfo, 'small', str(framecount))
        data_output(m_dfo, 'middle', str(framecount))
        data_output(l_dfo, 'large', str(framecount))
        data_output(p_dfo, 'poly', str(framecount))

    if framecount == 40 or framecount == 140 or framecount == 240:
        velocity_bubble = calculate_velocity(previous, memory)
        print(velocity_bubble)
        s_dfo, m_dfo, l_dfo, p_dfo = position_classify(memory, velocity_bubble)
        data_output(s_dfo, 'small', str(framecount))
        data_output(m_dfo, 'middle', str(framecount))
        data_output(l_dfo, 'large', str(framecount))
        data_output(p_dfo, 'poly', str(framecount))


cv.waitKey(0)
capture.release()
cv.destroyAllWindows()
