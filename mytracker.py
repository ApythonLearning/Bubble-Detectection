import numpy as np


def iou_meter(pre_bboxes, bboxes, threshold=0.3):
    """
    only use for a simple calculate the iou scores between frames which connected
    :param pre_bboxes: pre frame in a video
    :param bboxes:  current frame of a video
    :return: iou scores
    """

    score_array = np.zeros((len(pre_bboxes), len(bboxes)), dtype=np.float16)

    for i in range(len(pre_bboxes)):
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
                if iou_score < threshold:  # set the threshold of the iou scores, in line with the sort
                    iou_score = 0
                score_array[i][j] = iou_score

    return score_array


def distance_meter(pre_bboxes, bboxes, threshold=1):

    """
    calculating the score of the distance which calculated by finding out the
    Euclidean distance between one box and other boxes.In order to find the nearby
    boxes of the box.Then we can choose one which most similar to the target.
    :param pre_boxes:
    :param bboxes:
    :return:
    """
    dis_scores = np.zeros((len(pre_bboxes), len(bboxes)), dtype=np.float16)

    for i in range(len(pre_bboxes)):
        for j in range(len(bboxes)):

            (pre_x1, pre_y1) = (pre_bboxes[i][0], pre_bboxes[i][1])
            (pre_x2, pre_y2) = (pre_bboxes[i][2], pre_bboxes[i][3])
            (cur_x1, cur_y1) = (bboxes[j][0], bboxes[j][1])
            (cur_x2, cur_y2) = (bboxes[j][2], bboxes[j][3])
            origin_w = pre_x2 - pre_x1
            origin_h = pre_y2 - pre_y1
            current_w = cur_x2 - cur_x1
            current_h = cur_y2 - cur_y1
            pose_distance = ((origin_w + current_w)**2/4 + (origin_h + current_h)**2/4)**0.5

            (pre_ct_x, pre_ct_y) = (int((pre_x1+pre_x2)/2), int((pre_y1+pre_y2)/2))
            (cur_ct_x, cur_ct_y) = (int((cur_x1+cur_x2)/2), int((cur_y1+cur_y2)/2))

            ecu_distance = ((pre_ct_x - cur_ct_x)**2 + (pre_ct_y - cur_ct_y)**2)**0.5
            if ecu_distance == 0:
                ecu_distance = 0.001
            dis_ratio = pose_distance / ecu_distance
            if dis_ratio > threshold:
                dis_scores[i][j] = dis_ratio
                # dis_scores[i][j] = ecu_distance

    return dis_scores


def match_optimal(iou_prearray):
    """
    Designed for matching the boxes of a current frame with
    the boxes of a frame which before the current frame.
    :param iou_prearray:
    :return:
    """
    iou_array = iou_prearray.T
    # print(iou_array.shape)
    # print(iou_array)
    match_boxes = []
    match_index = np.argmax(iou_array, axis=1)  # find out the max iou score index
    match_index = match_index.tolist()
    equ_count = 0
    for i in range(len(match_index)-1):
        for j in range(i+1, len(match_index)):
            if match_index[j] == match_index[i]:
                equ_count += 1
                if iou_array[i][match_index[j]] > iou_array[i][match_index[i]]:
                    match_index[i] = iou_array.shape[1]+equ_count
                else:
                    if iou_array[i][match_index[j]] or iou_array[i][match_index[i]] == 0:
                        match_index[i] = iou_array.shape[1] + equ_count
                        equ_count += 1
                        match_index[j] = iou_array.shape[1] + equ_count
                    match_index[j] = iou_array.shape[1]+equ_count

    max_idx = max(match_index)
    for i in range(0, max_idx+1):
        if i in match_index:
            match_boxes.append(match_index.index(i))
        else:
            match_boxes.append('None')

    # print(match_boxes)
    return match_index, match_boxes


def match_optical(metric_array):
    """
    Designed for matching the boxes of a current frame with
    the boxes of a frame which before the current frame.
    :param metric_array:
    :return:
    """
    iou_array = metric_array.T
    print(iou_array.shape)
    print(iou_array)
    match_boxes = []
    match_index = np.argmin(iou_array, axis=1)  # find out the max iou score index
    match_index = match_index.tolist()
    print(match_index)
    equ_count = 0
    for i in range(len(match_index)-1):
        for j in range(i+1, len(match_index)):
            if match_index[j] == match_index[i]:
                equ_count += 1
                if iou_array[i][match_index[j]] < iou_array[i][match_index[i]]:
                    match_index[i] = iou_array.shape[1]+equ_count
                else:
                    if iou_array[i][match_index[j]] or iou_array[i][match_index[i]] == 0:
                        match_index[i] = iou_array.shape[1] + equ_count
                        equ_count += 1
                        match_index[j] = iou_array.shape[1] + equ_count
                    match_index[j] = iou_array.shape[1]+equ_count

    max_idx = max(match_index)
    for i in range(0, max_idx+1):
        if i in match_index:
            match_boxes.append(match_index.index(i))
        else:
            match_boxes.append('None')

    print(match_boxes)
    return match_index, match_boxes


def count_collision(iou_metric):
    """
    count number of the collision times which is in line with the matrix of pre_box to current_box;
    the split number is in line with the matrix of current to pre_box.
    :param iou_metric:the iou score which calculated by the iou_meter between pre box and current box.
    :return:the two output num of collision and split.
    """
    pre2cur = np.argmax(iou_metric, axis=1)
    cur2pre = np.argmax(iou_metric, axis=0)
    iou_metric_t = iou_metric.T

    equ_count_c = 0
    for i in range(len(pre2cur) - 1):
        for j in range(i + 1, len(pre2cur)):
            if pre2cur[j] == pre2cur[i]:
                if iou_metric[i][pre2cur[j]] or iou_metric[i][pre2cur[i]] != 0:
                    equ_count_c += 1

    equ_count_s = 0
    for i in range(len(cur2pre) - 1):
        for j in range(i + 1, len(cur2pre)):
            if cur2pre[j] == cur2pre[i]:
                if iou_metric_t[i][cur2pre[j]] or iou_metric_t[i][cur2pre[i]] != 0:
                    equ_count_s += 1

    output_num = (equ_count_c, equ_count_s)

    return output_num


class MyTracker:

    def __init__(self):

        self.trackers = []
        self.pre_trackers = []
        self.origin_box = []
        self.pre_origin_box = []
        self.frame_num = 0

    def load_boxes(self, boxes):
        """
        loading the parameter of bounding boxes
        :param boxes:
        :return:
        """

        if self.frame_num == 0:
            self.trackers = boxes
            self.origin_box = boxes
        else:
            self.pre_origin_box = self.origin_box
            self.pre_trackers = self.trackers
            self.trackers = boxes
            self.origin_box = boxes
        self.frame_num += 1
        return print('Current frame is: ' + str(self.frame_num + 2))

    def updates(self):
        """
        updating the iou scores of all the boxes in a frame
        :return:
        """
        if self.frame_num == 0:
            raise AttributeError('The first frame can not use for a iou calculating')

        iou_scores = iou_meter(self.pre_trackers, self.trackers, threshold=0.3)
        pre2cur, cur2pre = match_optimal(iou_scores)

        iou_metrics = iou_meter(self.pre_origin_box, self.origin_box, threshold=0.6)
        out_put_num = count_collision(iou_metrics)
        print(out_put_num)

        # dis_scores = distance_meter(self.pre_trackers, self.trackers)
        # pre2cur, cur2pre = match_optimal(dis_scores)

        sorting = []
        for pre_id, cur_id in enumerate(cur2pre):
            if cur_id != 'None':
                sorting.append(self.trackers[cur_id])
            else:
                sorting.append([0, 0, 0, 0])

        self.trackers = sorting

        return sorting, out_put_num









