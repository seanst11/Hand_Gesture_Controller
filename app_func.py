#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
import time
import math
import sys

from collections import Counter
from collections import deque
from gtts import gTTS
from playsound import playsound
from PyQt5.Qt import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QCursor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

from utils import CvFpsCalc

# models
from model import KeyPointClassifier_R
from model import KeyPointClassifier_L
from model import PointHistoryClassifier
from model import MouseClassifier

from func import *



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Window size
        self.WIDTH = 60
        self.HEIGHT = 60
        self.resize(self.WIDTH, self.HEIGHT)

        # Widget
        self.centralwidget = QWidget(self)
        self.centralwidget.resize(self.WIDTH, self.HEIGHT)

        # Menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.right_menu)

        # Initial
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.3)

        self.radius = 30
        self.centralwidget.setStyleSheet(
            """
            background:rgb(255, 255, 255);
            border-radius:{0}px;
            """.format(self.radius)
        )

        self.runButton = QPushButton(self)
        self.runButton.setText("Run")  # text
        self.runButton.setGeometry(10, 10, 40, 20)
        self.work = WorkThread()
        self.runButton.clicked.connect(self.execute)
        self.check_worked = False

    def execute(self):
        # 启动线程
        if self.check_worked == False:
            self.check_worked = True
            self.work.start()
            # 线程自定义信号连接的槽函数
            self.work.trigger.connect(self.display)
            self.runButton.setText('Stop')
            self.centralwidget.setStyleSheet(
                """
                background:rgb(255, 200, 255);
                border-radius:{0}px;
                """.format(self.radius))

        else:
            self.check_worked = False
            self.work.stop()
            self.runButton.setText('Run')

    def display(self, int):
        # 由于自定义信号时自动传递一个字符串参数，所以在这个槽函数中要接受一个参数
        # self.listWidget.addItem(str)
        if int == 0:
            # self.setStyleSheet("background-color: yellow;")
            self.centralwidget.setStyleSheet(
                """
                background:rgb(255, 0, 0);
                border-radius:{0}px;
                """.format(self.radius)
            )
        if int == 1:
            # self.setStyleSheet("background-color: blue;")
            self.centralwidget.setStyleSheet(
                """
                background:rgb(0, 255, 0);
                border-radius:{0}px;
                """.format(self.radius)
            )
        if int == 2:
            # self.setStyleSheet("background-color: green;")
            self.centralwidget.setStyleSheet(
                """
                background:rgb(0, 0, 255);
                border-radius:{0}px;
                """.format(self.radius)
            )

    def right_menu(self, pos):
        menu = QMenu()

        # Add menu options
        exit_option = menu.addAction('Exit')

        # Menu option events
        exit_option.triggered.connect(lambda: sys.exit())

        # Position
        menu.exec_(self.mapToGlobal(pos))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.moveFlag = True
            self.movePosition = event.globalPos() - self.pos()
            self.setCursor(QCursor(Qt.OpenHandCursor))
            event.accept()

    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self.moveFlag:
            self.move(event.globalPos() - self.movePosition)
            event.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.moveFlag = False
        self.setCursor(Qt.CrossCursor)



class WorkThread(QThread):
    # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    trigger = pyqtSignal(int)

    def __int__(self):
        # 初始化函数
        super(WorkThread, self).__init__()
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        # Argument parsing #################################################################
        self.stop_flag = False
        args = get_args()

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        use_brect = True

        # Camera preparation ###############################################################
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier_R = KeyPointClassifier_R(invalid_value=8, score_th=0.4)
        keypoint_classifier_L = KeyPointClassifier_L(invalid_value=8, score_th=0.4)
        mouse_classifier = MouseClassifier(invalid_value=2, score_th=0.4)
        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open(
                'model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=3)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)
        mouse_id_history = deque(maxlen=40)

        # 靜態手勢最常出現參數初始化
        keypoint_length = 5
        keypoint_R = deque(maxlen=keypoint_length)
        keypoint_L = deque(maxlen=keypoint_length)

        # result deque
        rest_length = 300
        rest_result = deque(maxlen=rest_length)
        speed_up_count = deque(maxlen=3)

        # ========= 使用者自訂姿勢、指令區 =========
        # time.sleep(0.5)
        # keepadd = False

        # ========= 按鍵前置作業 =========
        mode = 0
        presstime = presstime_2 = presstime_3 = resttime = presstime_4 = time.time()

        detect_mode = 2
        what_mode = 'mouse'
        landmark_list = 0
        pyautogui.PAUSE = 0

        # ========= 滑鼠前置作業 =========
        wScr, hScr = pyautogui.size()
        frameR = 100
        smoothening = 7
        plocX, plocY = 0, 0
        clocX, clocY = 0, 0
        mousespeed = 1.5
        clicktime = time.time()
        #關閉 滑鼠移至角落啟動保護措施
        pyautogui.FAILSAFE = False

        # ========= google 小姐 =========
        # speech_0 = gTTS(text="スリープモード", lang='ja')
        # speech_0.save('rest.mp3')
        # speech_0 = gTTS(text="キーボードモード", lang='ja')
        # speech_0.save('keyboard.mp3')
        # speech = gTTS(text="マウスモード", lang='ja')
        # speech.save('mouse.mp3')

        # ===============================
        i = 0
        finger_gesture_id = 0

        # ========= 主程式運作 =========
        while (not self.stop_flag):
            left_id = right_id = -1
            fps = cvFpsCalc.get()

            # Process Key (ESC: end)
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Camera capture
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True



            ####rest_result####
            if results.multi_hand_landmarks is None:
                rest_id = 0
                rest_result.append(rest_id)
            if results.multi_hand_landmarks is not None:
                rest_id = 1
                rest_result.append(rest_id)
            most_common_rest_result = Counter(rest_result).most_common()

            # old version for 10 sec to rest mode####################
            #print(most_common_rest_result[0])
            # if most_common_rest_result[0][0] == 0 and most_common_rest_result[0][1] == 300:
            #     if detect_mode != 0:
            #
            #         print('Mode has changed')
            #         detect_mode = 0
            #         what_mode = 'Rest'
            #         print(f'Current mode => {what_mode}')

            # new version for 10 sec to rest mode###################
            if time.time() - resttime > 10:
                if detect_mode != 0:
                    detect_mode = 0
                    what_mode = 'Sleep'
                    print(f'Current mode => {what_mode}')

            ####rest_result####




            #  ####################################################################
            # print(most_common_rest_result)
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # print(landmark_list)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                    # 靜態手勢資料預測
                    hand_sign_id_R = keypoint_classifier_R(pre_processed_landmark_list)
                    hand_sign_id_L = keypoint_classifier_L(pre_processed_landmark_list)
                    mouse_id = mouse_classifier(pre_processed_landmark_list)
                    # print(mouse_id)
                    if handedness.classification[0].label[0:] == 'Left':
                        left_id = hand_sign_id_L

                    else:
                        right_id = hand_sign_id_R

                    # 手比one 觸發動態資料抓取
                    if right_id == 1 or left_id ==1:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # 動態手勢資料預測
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    # print(finger_gesture_id) # 0 = stop, 1 = clockwise, 2 = counterclockwise, 3 = move,偵測出現的動態手勢

                    # 動態手勢最常出現id #########################################
                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    #滑鼠的deque
                    mouse_id_history.append(mouse_id)
                    most_common_ms_id = Counter(mouse_id_history).most_common()
                    # print(f'finger_gesture_history = {finger_gesture_history}')
                    # print(f'most_common_fg_id = {most_common_fg_id}')

                    # 靜態手勢最常出現id #########################################
                    hand_gesture_id = [right_id, left_id]
                    keypoint_R.append(hand_gesture_id[0])
                    keypoint_L.append(hand_gesture_id[1])
                    # print(keypoint_R) # deque右手的靜態id
                    # print(most_common_keypoint_id) # 右手靜態id最大
                    if right_id != -1:
                        most_common_keypoint_id = Counter(keypoint_R).most_common()
                    else:
                        most_common_keypoint_id = Counter(keypoint_L).most_common()



                    # print(f'keypoint = {keypoint}')
                    # print(f'most_common_keypoint_id = {most_common_keypoint_id}')

                    ###############################################################

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[most_common_keypoint_id[0][0]],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
                    resttime = time.time()
            else:
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # 偵測是否有手勢 #########################################

            if left_id + right_id > -2:
                if time.time() - presstime > 1:
                    # change mode
                    if most_common_ms_id[0][0] == 3 and most_common_ms_id[0][1] == 40: #Gesture six changes to the different mode
                        print('Mode has changed')
                        detect_mode = (detect_mode + 1) % 3
                        if detect_mode == 0:
                            what_mode = 'Sleep'
                            playsound('rest.mp3', block=False)
                        if detect_mode == 1:
                            what_mode = 'Keyboard'
                            playsound('keyboard.mp3', block=False)
                        if detect_mode == 2:
                            what_mode = 'Mouse'
                            playsound('mouse.mp3', block=False)
                        print(f'Current mode => {what_mode}')
                        presstime = time.time() + 1

                    # control keyboard
                    elif detect_mode == 1:
                        if time.time() - presstime_2 > 1:
                            # 靜態手勢控制
                            control_keyboard(most_common_keypoint_id, 2, 'K', keyboard_TF=True, print_TF=False)
                            # control_keyboard(most_common_keypoint_id, 0, 'right', keyboard_TF=True, print_TF=False)
                            # control_keyboard(most_common_keypoint_id, 7, 'left', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 9, 'C', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 5, 'up', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 6, 'down', keyboard_TF=True, print_TF=False)
                            presstime_2 = time.time()

                        # right右鍵
                        if most_common_keypoint_id[0][0] == 0 and most_common_keypoint_id[0][1] == 5:
                            # print(i, time.time() - presstime_4)
                            if i == 3 and time.time() - presstime_4 > 0.3:
                                pyautogui.press('right')
                                i = 0
                                presstime_4 = time.time()
                            elif i == 3 and time.time() - presstime_4 > 0.25:
                                pyautogui.press('right')
                                presstime_4 = time.time()
                            elif time.time() - presstime_4 > 1:
                                pyautogui.press('right')
                                i += 1
                                presstime_4 = time.time()

                        # left左鍵
                        if most_common_keypoint_id[0][0] == 7 and most_common_keypoint_id[0][1] == 5:
                            # print(i, time.time() - presstime_4)
                            if i == 3 and time.time() - presstime_4 > 0.3:
                                pyautogui.press('left')
                                i = 0
                                presstime_4 = time.time()
                            elif i == 3 and time.time() - presstime_4 > 0.25:
                                pyautogui.press('left')
                                presstime_4 = time.time()
                            elif time.time() - presstime_4 > 1:
                                pyautogui.press('left')
                                i += 1
                                presstime_4 = time.time()

                        # 動態手勢控制
                        if most_common_fg_id[0][0] == 1 and most_common_fg_id[0][1] > 12:
                            if time.time() - presstime_3 > 1.5:
                                #pyautogui.press(['shift', '>'])
                                pyautogui.hotkey('shift', '>')
                                print('speed up')
                                presstime_3 = time.time()
                        elif most_common_fg_id[0][0] == 2 and most_common_fg_id[0][1] > 12:
                            if time.time() - presstime_3 > 1.5:
                                #pyautogui.press(['shift', '<'])
                                pyautogui.hotkey('shift', '<')
                                print('speed down')
                                presstime_3 = time.time()


                if detect_mode == 2:
                    if mouse_id == 0:  # Point gesture
                        # print(landmark_list[8]) #index finger
                        # print(landmark_list[12]) #middle finger
                        x1, y1 = landmark_list[8]
                        # cv.rectangle(debug_image, (frameR, frameR), (cap_width - frameR, cap_height - frameR),
                        #              (255, 0, 255), 2)
                        cv.rectangle(debug_image, (50, 30), (cap_width - 50, cap_height - 170),
                                     (255, 0, 255), 2)
                        #座標轉換
                        #x軸: 鏡頭上50~(cap_width - 50)轉至螢幕寬0~wScr
                        #y軸: 鏡頭上30~(cap_height - 170)轉至螢幕長0~hScr
                        x3 = np.interp(x1, (50, (cap_width - 50)), (0, wScr))
                        y3 = np.interp(y1, (30, (cap_height - 170)), (0, hScr))
                        # print(x3, y3)

                        # 6. Smoothen Values
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        # 7. Move Mouse
                        pyautogui.moveTo(clocX, clocY)
                        cv.circle(debug_image, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                        plocX, plocY = clocX, clocY

                    if mouse_id == 1:
                        length, img, lineInfo = findDistance(landmark_list[8], landmark_list[12], debug_image)

                        # 10. Click mouse if distance short
                        if time.time() - clicktime > 0.5:
                            if length < 40:
                                cv.circle(img, (lineInfo[4], lineInfo[5]),
                                          15, (0, 255, 0), cv.FILLED)
                                pyautogui.click()
                                print('click')
                                clicktime = time.time()


                            # if length > 70:
                            #     cv.circle(img, (lineInfo[4], lineInfo[5]),
                            #               15, (0, 255, 0), cv.FILLED)
                                # pyautogui.click(clicks=2)
                                # print('click*2')
                                # clicktime = time.time()

                    if most_common_keypoint_id[0][0] == 5 and most_common_keypoint_id[0][1] == 5:
                        pyautogui.scroll(20)

                    if most_common_keypoint_id[0][0] == 6 and most_common_keypoint_id[0][1] == 5:
                        pyautogui.scroll(-20)

                    #if left_id == 7 or right_id == 7:
                    if most_common_keypoint_id[0][0] == 0 and most_common_keypoint_id[0][1] == 5:
                        if time.time() - clicktime > 1:
                            pyautogui.click(clicks=2)
                            clicktime = time.time()

                    if most_common_keypoint_id[0][0] == 9 and most_common_keypoint_id[0][1] == 5:
                        if time.time() - clicktime > 2:
                            pyautogui.hotkey('alt', 'left')
                            clicktime = time.time()

            cv.putText(debug_image, what_mode, (400, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            # Screen reflection ###################################JL##########################
            cv.imshow('Hand Gesture Recognition', debug_image)

            self.trigger.emit(detect_mode)

        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())