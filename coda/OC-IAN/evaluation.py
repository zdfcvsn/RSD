import os
import cv2
import collections
import numpy as np

dataset = 'Type-I'
models= ['OC_IAN']
for model in models:
    print(model)
    truth_path = './../../data/RSDDs/'+dataset+'/test/GroundTruth'
    result_path = './result/'+dataset+'/'+model+'/'

    num = 0
    avg_R = 0
    avg_P = 0
    avg_F1 = 0
    for img in os.listdir(truth_path):
        num = num + 1
        ptruth = os.path.join(truth_path,img)
        presult = os.path.join(result_path,img)
        truth = cv2.imread(ptruth, 0)
        result = cv2.imread(presult, 0)

        label = truth[:, 0]
        label[label <= 130] = 0
        label[label > 130] = 2
        pred = result[:, 0]
        pred[pred <= 130] = 0
        pred[pred > 130] = 2

        label =np.asarray(label, dtype=np.int16)
        pred = np.asarray(pred, dtype=np.int16)
        diff_pl = pred-label
        G = collections.Counter(label)
        # print(G)
        F = G[0]
        T = G[2]
        D = collections.Counter(diff_pl)
        FN = D[-2]
        FP = D[2]
        TP = T-FN
        TN = F-FP

        if TP+FP != 0 and TP != 0:
            R = TP / T
            P = TP/(TP+FP)
            avg_R = avg_R + R
            avg_P = avg_P + P

    avg_R = avg_R/num
    avg_P = avg_P/num
    avg_F1 = 2*avg_R*avg_P/(avg_R+avg_P)
    print('line-level - avg_P:{}, avg_R:{}, avg_F1:{}'.format(avg_P, avg_R, avg_F1))

    # **-----------------------------------------
    num = 0
    avg_R = 0
    avg_P = 0
    avg_F1 = 0
    for img in os.listdir(truth_path):
        num = num + 1
        ptruth = os.path.join(truth_path,img)
        presult = os.path.join(result_path,img)
        truth = cv2.imread(ptruth, 0)
        result = cv2.imread(presult, 0)

        label = truth[:, 0]
        label[label <= 130] = 0
        label[label > 130] = 2
        pred = result[:, 0]
        pred[pred <= 130] = 0
        pred[pred > 130] = 2
        label =np.asarray(label, dtype=np.int16)
        pred = np.asarray(pred, dtype=np.int16)
    ###-------------------------------------------------
        t_id = []
        t_block = []
        TP = 0
        defect_num = 0
        for i in range(len(label)-1):
            if label[i] == 2:
                if len(t_id) > 0 and t_id[-1] + 1 != i:
                    defect_num = defect_num + 1
                t_id.append(i)
                t_block.append(defect_num)

        T = len(np.unique(t_block))

        for i in np.unique(t_block):
            local = np.where(t_block == i)
            # print(local)
            for j in local[0]:
                if pred[t_id[j]] == 2:
                    TP = TP + 1
                    break

    ###---------------------------------------------------
        p_id = []
        p_block = []
        pred_num = 0

        for i in range(len(pred) - 1):
            if pred[i] == 2:
                if len(p_id) > 0 and p_id[-1] + 1 != i:
                    pred_num = pred_num + 1
                p_id.append(i)
                p_block.append(pred_num)

        pred_num = len(np.unique(p_block))

        # for i in np.unique(p_block):
        #     local = np.where(p_block == i)
        #     # print(local[0])
        #     for j in local[0]:
        #         if label[p_id[j]] == 2:
        #             TP = TP + 1
        #             break

        if pred_num != 0 and TP != 0:
            R = TP / T
            P = TP/pred_num
        else:
            R = 0
            P = 0
        avg_R = avg_R + R
        avg_P = avg_P + P
        # avg_F1 = avg_F1 + F1
        # print('{} {} {}  '.format(img, P, R))
        # print('------------------------')
    print(avg_R)
    avg_R = avg_R/num
    avg_P = avg_P/num
    avg_F1 = 2*avg_R*avg_P/(avg_R+avg_P)

    print('defect-level - avg_P:{}, avg_R:{}, avg_F1:{}'.format(avg_P, avg_R, avg_F1))
    print('\n')
