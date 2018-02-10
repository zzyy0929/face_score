# -- coding: utf-8 --
#   操作方法：把照片放在/home/zzy/face_recognition_image/下 分别建立文件夹（可以是正常的照片），修改name=[]数组对应人的姓名
#   修改video_capture = cv2.VideoCapture("/home/zzy/yawn_data/szc.avi")测试视频还是实时检测

import sys
import os
import dlib
import glob
from skimage import io
import csv
import numpy as np
import cv2
# import pyttsx
from datetime import datetime
# import socket
from scipy.spatial.distance import pdist

# address=('192.168.1.107',8085)   #set the address
# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
order = ["zzy", "szc", "zh", "jxn", "zjw", "zrj", "gyq", "Unknown"]


def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataMat = [];
    labelMat = [];
    labelMat_1 = []  # label_1 是性别，倒数第二列，labelmat是评分最后一列
    dataset = list(lines)
    # for i in range(len(dataset)):
    #     dataset[i] = [float(x) for x in dataset[i]]
    for i in range(len(dataset)):
        lineArr = []
        vector = dataset[i]
        labelMat.append(float(vector[-1]))
        # labelMat_1.append(float(vector[128]))
        # print(labelMat_1)
        for j in range(0, 128):
            lineArr.append(float(vector[j]))
        dataMat.append(lineArr)
    return dataMat, labelMat


def euler_dist(vector1, vector2):
    X = np.vstack([vector1, vector2])
    dist = pdist(X)
    return dist


def cos(vector1, vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


if __name__ == '__main__':
    # engine = pyttsx.init()

    [dataMat, labelMat] = loadCsv("/home/zzy/face_rank/score_train.csv")
    predictor_path = "/home/zzy/dlib-19.6/shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "/home/zzy/dlib-19.6/dlib_face_recognition_resnet_model_v1.dat"
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    labelMat = list(labelMat)

    total_image = np.shape(labelMat)[0]
    score = [95, 90, 85, 80, 70, 65]
    #     用视频测试
    # video_capture = cv2.VideoCapture("/home/zzy/yawn_data/fyp.avi")
    # video_capture = cv2.VideoCapture("/home/zzy/jilu_record/gyq/gyq.avi")
    # video_capture = cv2.VideoCapture(0)
    # while (True):
    #     starttime = datetime.now()
    #     ret, frame = video_capture.read()  # Read the image with OpenCV
    #     dets = detector(frame, 1)
    #     # if dets is None:
    #     #     raise Exception("Unable to align the frame")
    #     for k, d in enumerate(dets):
    #         shape = sp(frame, dets[0])
    #     name_index=-1
    #     try:
    #         face_descriptor = facerec.compute_face_descriptor(frame, shape)
    #         face_descriptor_trans = np.transpose(face_descriptor)
    #         dist=np.zeros(total_image)
    #         dist_euler=np.zeros(total_image)
    #         for i in range(0,total_image):
    #             dist[i]=1-cos(face_descriptor,dataMat[i])   #两个距离 dist[i]和dist_euler
    #             dist_euler[i]=euler_dist(face_descriptor,dataMat[i])
    #         dist1=dist.tolist()
    #         dist2=dist_euler.tolist()
    #
    #         new_dist1=sorted(dist1)
    #         new_dist2=sorted(dist2)
    #
    #         loca_dist1=np.zeros(3);score_index_1=np.zeros(3)
    #         loca_dist2=np.zeros(3);score_index_2=np.zeros(3)
    #
    #         for j in range(0,3):
    #             loca_dist1[j]=dist1.index(new_dist1[j])
    #             score_index_1[j]=np.uint8(labelMat[loca_dist1[j]])
    #             print(score[name_index_1[j]])
    #
    #
    #         #break
    #     except:
    #         print('Are you sure you have the face?!')
    #     #cv2.putText(frame,"Hello,"+str(score[name_index]+"!") , (400, 30), cv2.FONT_HERSHEY_SIMPLEX,
    # 1.0, (0, 255, 0), 2)
    #     cv2.imshow("face_recog", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'): break

    #     用图片测试
    for kk in range(2, 13):
        frame = cv2.imread("/home/zzy/face_rank/face_score_test/" + str(kk) + ".jpg")
        height = np.shape(frame)[0]
        width = np.shape(frame)[1]
        dets = detector(frame, 1)
        for k, d in enumerate(dets):
            shape = sp(frame, dets[0])
        name_index = -1
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        dataMat_trans = np.transpose(dataMat)
        pi_a = np.linalg.pinv(dataMat_trans)
        s = pi_a.dot(face_descriptor) #作为方程的解
        print(s)
        sum_s=0.0
        for j in range(0,total_image):
            sum_s=sum_s+s[j]
        print(sum_s)
        w=np.zeros(total_image)
        for j in range(0,total_image):
            w[j]=s[j]/sum_s

        print(w)
        w1=w
        w=w.tolist()
        new_w=sorted(w1)
        print(new_w)
        a=np.shape(w)[0]
        print(a)
        select_num=5
        index_1=np.zeros(select_num)
        class1=np.zeros(select_num)
        for j in range(0, select_num):
            index_1[j]=w.index(new_w[a-1-j])
            class1[j]=labelMat[np.uint8(index_1[j])]

        final_score1 = score[np.uint8(class1[0])] * 0.333 + score[np.uint8(class1[1])] * 0.333 + score[
            np.uint8(class1[2])] * 0.333
        print(final_score1)

        face_descriptor_trans = np.transpose(face_descriptor)
        dist = np.zeros(total_image)
        dist_euler = np.zeros(total_image)
        for i in range(0, total_image):
            dist[i] = 1 - cos(face_descriptor, dataMat[i])  # 两个距离 dist[i]和dist_euler
            dist_euler[i] = euler_dist(face_descriptor, dataMat[i])

        dist1 = list(dist)
        dist2 = list(dist_euler)
        new_dist1 = sorted(dist)
        new_dist2 = sorted(dist)
        loca_dist1 = np.zeros(3)
        score_1 = np.zeros(3)
        loca_dist2 = np.zeros(3)
        score_2 = np.zeros(3)
        for j in range(0, 3):
            loca_dist1[j] = dist1.index(new_dist1[j])
            score_1[j] = labelMat[np.uint8(loca_dist1[j])]
            loca_dist2[j] = dist1.index(new_dist1[j])
            score_2[j] = labelMat[np.uint8(loca_dist2[j])]
        final_score = score[np.uint8(score_1[0])] * 0.333 + score[np.uint8(score_1[1])] * 0.333 + score[
            np.uint8(score_1[2])] * 0.333
        final_score=final_score1*0.6+final_score*0.4


























        text = "Your score is " + str(final_score)
        print(text)


        cv2.putText(frame, text, (width / 5 * 1, height / 5 * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow("face_rank", frame)
        cv2.waitKey(0)
