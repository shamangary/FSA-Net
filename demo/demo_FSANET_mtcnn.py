import os
import cv2
import sys
sys.path.append('..')


import numpy as np
from math import cos, sin
# from moviepy.editor import *
from lib.FSANET_model import *
from mtcnn.mtcnn import MTCNN

from keras import backend as K
from keras.layers import Average
from keras.models import Model


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                 * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                 * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def draw_results_mtcnn(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot):

    if len(detected) > 0:
        for i, d in enumerate(detected):
            #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            if d['confidence'] > 0.95:
                x1, y1, w, h = d['box']

                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                faces[i, :, :, :] = cv2.resize(
                    input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = cv2.normalize(
                    faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                p_result = model.predict(face)

                face = face.squeeze()
                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :],
                                p_result[0][0], p_result[0][1], p_result[0][2])

                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img

    cv2.imshow("result", input_img)
    
    return input_img  # ,time_network,time_plot


def main():
    try:
        os.mkdir('./img')
    except OSError:
        pass

    K.set_learning_phase(0)  # make sure its testing mode
    # face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    detector = MTCNN()

    # load model and weights
    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    detected = ''  # make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
    ad = 0.6

    # Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes,
                             stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    print('Loading models ...')

    weight_file1 = '../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')

    weight_file2 = '../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    # capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)

    print('Start detecting pose ...')
    detected_pre = []

    while True:
        # get video frame
        ret, input_img = cap.read()

        img_idx = img_idx + 1
        img_h, img_w, _ = np.shape(input_img)

        if img_idx == 1 or img_idx % skip_frame == 0:
            time_detection = 0
            time_network = 0
            time_plot = 0

            # detect faces using LBP detector
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # detected = face_cascade.detectMultiScale(gray_img, 1.1)
            detected = detector.detect_faces(input_img)

            if len(detected_pre) > 0 and len(detected) == 0:
                detected = detected_pre

            faces = np.empty((len(detected), img_size, img_size, 3))

            input_img = draw_results_mtcnn(
                detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot)
            cv2.imwrite('img/'+str(img_idx)+'.png', input_img)

        else:
            input_img = draw_results_mtcnn(
                detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot)

        if len(detected) > len(detected_pre) or img_idx % (skip_frame*3) == 0:
            detected_pre = detected

        key = cv2.waitKey(1)


if __name__ == '__main__':
    main()
