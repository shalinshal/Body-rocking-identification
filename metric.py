# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:35:10 2019

@author: Shalin
"""

#inputs are "maps" which are just a list of lists. Please ignore the poor naming convention.
#Each index in the list represents an event or detection interval.
#map[i][0] is the start index of the ith event/detection.
#Similarly map[i][1] is the end index of the ith event/detection
import os
import numpy as np
def get_metric_values(gt_map,pred_map):
    gt_idx = 0
    pred_idx = 0
    while gt_idx < len(gt_map) and pred_idx < len(pred_map):
        if gt_map[gt_idx][0] <= pred_map[pred_idx][0]:
            if gt_map[gt_idx][1] < pred_map[pred_idx][0]:
                FN += 1
                gt_idx += 1
            elif gt_map[gt_idx][1] >= pred_map[pred_idx][1]:
                TP += 1
                gt_idx, pred_idx = increment_idcs_exit_event(gt_map, pred_map, gt_idx, pred_idx)
            elif get_overlap(gt_map[gt_idx],pred_map[pred_idx]) < 0.5:
                FN += 1
                gt_idx += 1
            else:
                TP += 1
                gt_idx += 1
                pred_idx += 1
        else:
            if gt_map[gt_idx][0] > pred_map[pred_idx][1]:
                FP += 1
                pred_idx += 1
            elif get_overlap(gt_map[gt_idx],pred_map[pred_idx]) < 0.5:
                FP += 1
                pred_idx += 1
            else:
                TP += 1
                gt_idx, pred_idx = increment_idcs_exit_event(gt_map, pred_map, gt_idx, pred_idx)
    FN += (len(gt_map) - gt_idx)
    FP += (len(pred_map) - pred_idx)
    return FP, FN, TP
    
    
def increment_idcs_exit_event(gt_map, pred_map, gt_idx, pred_idx):
    while gt_map[gt_idx][1] > pred_map[pred_idx][0]:
        if get_overlap(gt_map[gt_idx],pred_map[pred_idx]) < 0.5:
            gt_idx += 1
            return gt_idx, pred_idx
        pred_idx += 1
        if gt_idx >= len(gt_map) or pred_idx >= len(pred_map):
            return gt_idx, pred_idx
    gt_idx += 1
    return gt_idx, pred_idx
    
    
def get_overlap(gt_tup, pred_tup):
    if pred_tup[1] > gt_tup[1]:
        num = (gt_tup[1] - pred_tup[0])+1
    else:
        num = (pred_tup[1] - gt_tup[0])+1
    return num / ((pred_tup[1] - pred_tup[0])+1)

def get_maps(detections):
    _map = []
    for l in detections:
        counts = [0, 0, None] if int(l) == 1 else [None, None, None]
        for i, v in l:
            val = int(v)
            if val == 0:
                if all([counts[0], counts[1]]):
                    counts[2] = i
                    _map.append(counts)
                    counts = [None, None, None]
                    continue
            if counts[1] == (i - 1):
                counts[1] += 1
            else:
                counts = [i + 1, i, None]
        if all([counts[0], counts[1]]):
            counts[2] = len(l) - 1
            _map.append(counts)
    return list(map(lambda x: [x[0], x[2]], _map))

if __name__ == '__main__':
    label_path = 'D:\\NCSU\\sem2\\ECE 542 Neural Networks\\project\\3b\\ECE542-Proj3b\\Training Data B'
    predict_file_name = 'prediction.txt'
#    detection = [[0]*10 + [1]*5 + [0]*10 ]
#    print(detection)
#    print(get_maps(detection))

#    eval_obj = Eval()
    FP, FN, TP={},{},{}
    session_list = [13,16]
    for session_id in session_list:
        cur_label_path = os.path.join(label_path, 'Session{:02d}'.format(session_id), 'detection.txt')
        cur_predict_path = os.path.join(label_path, 'Session{:02d}'.format(session_id), predict_file_name)
        label_file = np.loadtxt(cur_label_path) 
        predict_file = np.loadtxt(cur_predict_path)
        FP[session_id], FN[session_id], TP[session_id] = get_metric_values(get_maps(label_file),get_maps(predict_file))

#    eval_obj.print_eval()