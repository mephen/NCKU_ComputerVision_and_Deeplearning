# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from hw1_process import CameraCalibration, LoadImage, AugmentedReality, StereoDisparityMap, SIFT, VGG19


camera_calibration = CameraCalibration()
load_image = LoadImage()
augmentedReality = AugmentedReality()
stereoDisparityMap = StereoDisparityMap()
sift = SIFT()
vgg19 = VGG19()


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setEnabled(True)
        Form.resize(936, 835)
        Form.setWindowOpacity(4.0)
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(250, 20, 181, 311))
        self.groupBox_2.setAutoFillBackground(True)
        self.groupBox_2.setObjectName("groupBox_2")


        self.find_distortion = QtWidgets.QPushButton(self.groupBox_2)
        self.find_distortion.setGeometry(QtCore.QRect(10, 220, 151, 31))
        self.find_distortion.setObjectName("find_distortion")
        self.find_distortion.clicked.connect(camera_calibration.find_distortion_and_show)


        self.find_intrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.find_intrinsic.setGeometry(QtCore.QRect(10, 70, 151, 31))
        self.find_intrinsic.setObjectName("find_intrinsic")
        self.find_intrinsic.clicked.connect(camera_calibration.find_intrinsic_and_show)


        self.show_result = QtWidgets.QPushButton(self.groupBox_2)
        self.show_result.setGeometry(QtCore.QRect(10, 260, 151, 31))
        self.show_result.setObjectName("show_result")
        self.show_result.clicked.connect(camera_calibration.show_undistorted)


        self.find_corners = QtWidgets.QPushButton(self.groupBox_2)
        self.find_corners.setGeometry(QtCore.QRect(10, 20, 151, 31))
        self.find_corners.setObjectName("find_corners")
        self.find_corners.clicked.connect(camera_calibration.find_corners_and_show_pictures)

        
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 110, 161, 91))
        self.groupBox_3.setAutoFillBackground(True)
        self.groupBox_3.setObjectName("groupBox_3")


        self.spinBox = QtWidgets.QSpinBox(self.groupBox_3)
        self.spinBox.setGeometry(QtCore.QRect(50, 20, 42, 22))
        self.spinBox.setObjectName("spinBox")


        self.find_extrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.find_extrinsic.setGeometry(QtCore.QRect(20, 160, 141, 31))
        self.find_extrinsic.setObjectName("find_extrinsic")
        self.find_extrinsic.clicked.connect(lambda: camera_calibration.find_extrinsic_and_show(index=self.spinBox.value()))


        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 201, 311))
        self.groupBox.setAutoFillBackground(True)
        self.groupBox.setObjectName("groupBox")


        self.load_folder = QtWidgets.QPushButton(self.groupBox)
        self.load_folder.setGeometry(QtCore.QRect(20, 60, 151, 41))
        self.load_folder.setObjectName("load_folder")
        self.load_folder.clicked.connect(load_image.load_folder)


        self.load_image_R = QtWidgets.QPushButton(self.groupBox)
        self.load_image_R.setGeometry(QtCore.QRect(20, 220, 151, 41))
        self.load_image_R.setObjectName("load_image_R")
        self.load_image_R.clicked.connect(load_image.load_image_R)


        self.load_image_L = QtWidgets.QPushButton(self.groupBox)
        self.load_image_L.setGeometry(QtCore.QRect(20, 140, 151, 41))
        self.load_image_L.setObjectName("load_image_L")
        self.load_image_L.clicked.connect(load_image.load_image_L)


        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(460, 20, 201, 311))
        self.groupBox_4.setAutoFillBackground(True)
        self.groupBox_4.setObjectName("groupBox_4")


        self.show_words_on_board = QtWidgets.QPushButton(self.groupBox_4)
        self.show_words_on_board.setGeometry(QtCore.QRect(10, 160, 181, 31))
        self.show_words_on_board.setObjectName("show_words_on_board")
        self.show_words_on_board.clicked.connect(lambda:augmentedReality.show_words_horizontal(words=self.ar_textEdit.toPlainText()))


        self.show_words_vertical = QtWidgets.QPushButton(self.groupBox_4)
        self.show_words_vertical.setGeometry(QtCore.QRect(10, 200, 181, 31))
        self.show_words_vertical.setObjectName("show_words_vertical")
        self.show_words_vertical.clicked.connect(lambda:augmentedReality.show_words_vertical(words=self.ar_textEdit.toPlainText()))


        self.ar_textEdit = QtWidgets.QTextEdit(self.groupBox_4)
        self.ar_textEdit.setGeometry(QtCore.QRect(20, 50, 161, 81))
        self.ar_textEdit.setFrameShape(QtWidgets.QFrame.Box)
        self.ar_textEdit.setObjectName("ar_textEdit")


        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setGeometry(QtCore.QRect(680, 20, 211, 311))
        self.groupBox_5.setAutoFillBackground(True)
        self.groupBox_5.setObjectName("groupBox_5")


        self.stereo_disparity_map = QtWidgets.QPushButton(self.groupBox_5)
        self.stereo_disparity_map.setGeometry(QtCore.QRect(10, 130, 191, 41))
        self.stereo_disparity_map.setObjectName("stereo_disparity_map")
        self.stereo_disparity_map.clicked.connect(stereoDisparityMap.stereo_disparity_map)


        self.groupBox_6 = QtWidgets.QGroupBox(Form)
        self.groupBox_6.setGeometry(QtCore.QRect(460, 350, 251, 400))
        self.groupBox_6.setAutoFillBackground(True)
        self.groupBox_6.setObjectName("groupBox_6")


        self.vgg19_load_image = QtWidgets.QPushButton(self.groupBox_6)
        self.vgg19_load_image.setGeometry(QtCore.QRect(20, 20, 191, 21))
        self.vgg19_load_image.setObjectName("load_image")
        self.vgg19_load_image.clicked.connect(lambda:vgg19.load_img(gui_object = self))

        self.vgg19_show_ar_images = QtWidgets.QPushButton(self.groupBox_6)
        self.vgg19_show_ar_images.setGeometry(QtCore.QRect(20, 50, 191, 61))
        self.vgg19_show_ar_images.setCheckable(False)
        self.vgg19_show_ar_images.setAutoExclusive(False)
        self.vgg19_show_ar_images.setAutoDefault(False)
        self.vgg19_show_ar_images.setObjectName("vgg19_show_ar_images")
        self.vgg19_show_ar_images.clicked.connect(vgg19.show_augmented_images)


        self.show_model_structure = QtWidgets.QPushButton(self.groupBox_6)
        self.show_model_structure.setGeometry(QtCore.QRect(20, 120, 191, 31))
        self.show_model_structure.setObjectName("show_model_structure")
        self.show_model_structure.clicked.connect(vgg19.show_moddel_structure)


        self.show_acc_and_loss = QtWidgets.QPushButton(self.groupBox_6)
        self.show_acc_and_loss.setGeometry(QtCore.QRect(20, 160, 191, 31))
        self.show_acc_and_loss.setObjectName("show_acc_and_loss")
        self.show_acc_and_loss.clicked.connect(vgg19.show_acc_and_loss)

        self.inference = QtWidgets.QPushButton(self.groupBox_6)
        self.inference.setGeometry(QtCore.QRect(20, 200, 191, 31))
        self.inference.setObjectName("inference")
        self.inference.clicked.connect(vgg19.show_inference)

        self.vgg19_graphicsView  = QtWidgets.QGraphicsView(self.groupBox_6)
        self.vgg19_graphicsView .setGeometry(QtCore.QRect(20, 260, 191, 128))
        self.vgg19_graphicsView .setFrameShape(QtWidgets.QFrame.Box)
        self.vgg19_graphicsView .setObjectName("vgg19_graphicsView ")
        self.scene = QGraphicsScene()
        self.vgg19_graphicsView.setScene(self.scene)


        self.label = QtWidgets.QLabel(self.groupBox_6)
        self.label.setGeometry(QtCore.QRect(20, 240, 47, 12))
        self.label.setObjectName("vgg19_label")


        self.groupBox_7 = QtWidgets.QGroupBox(Form)
        self.groupBox_7.setGeometry(QtCore.QRect(250, 350, 191, 241))
        self.groupBox_7.setAutoFillBackground(True)
        self.groupBox_7.setObjectName("groupBox_7")


        self.load_image1 = QtWidgets.QPushButton(self.groupBox_7)
        self.load_image1.setGeometry(QtCore.QRect(10, 40, 171, 31))
        self.load_image1.setObjectName("load_image1")
        self.load_image1.clicked.connect(sift.load_image1)


        self.load_image2 = QtWidgets.QPushButton(self.groupBox_7)
        self.load_image2.setGeometry(QtCore.QRect(10, 90, 171, 31))
        self.load_image2.setObjectName("load_image2")
        self.load_image2.clicked.connect(sift.load_image2)


        self.keypoints = QtWidgets.QPushButton(self.groupBox_7)
        self.keypoints.setGeometry(QtCore.QRect(10, 140, 171, 31))
        self.keypoints.setObjectName("keypoints")
        self.keypoints.clicked.connect(sift.key_points)


        self.matched_keypoints = QtWidgets.QPushButton(self.groupBox_7)
        self.matched_keypoints.setGeometry(QtCore.QRect(10, 180, 171, 31))
        self.matched_keypoints.setObjectName("matched_keypoints")
        self.matched_keypoints.clicked.connect(sift.matched_keypoints)


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_2.setTitle(_translate("Form", "1. Calibration"))
        self.find_distortion.setText(_translate("Form", "1.4 Find distortion"))
        self.find_intrinsic.setText(_translate("Form", "1.2 Find intrinsic"))
        self.show_result.setText(_translate("Form", "1.5 show result"))
        self.find_corners.setText(_translate("Form", "1.1 Find corners"))
        self.groupBox_3.setTitle(_translate("Form", "1.3 Find extrinsic"))
        self.find_extrinsic.setText(_translate("Form", "1.3 Find extrinsic"))
        self.groupBox.setTitle(_translate("Form", "Load Image"))
        self.load_folder.setText(_translate("Form", "Load folder"))
        self.load_image_R.setText(_translate("Form", "Load Image_R"))
        self.load_image_L.setText(_translate("Form", "Load Image_L"))
        self.groupBox_4.setTitle(_translate("Form", "2. Augmented Readlity"))
        self.show_words_on_board.setText(_translate("Form", "2.1 show words on board"))
        self.show_words_vertical.setText(_translate("Form", "2.2 show words vertical"))
        self.groupBox_5.setTitle(_translate("Form", "3. Stereo disparity map"))
        self.stereo_disparity_map.setText(_translate("Form", "3.1 stereo disparity map"))
        self.groupBox_6.setTitle(_translate("Form", "5. VGG19"))
        self.vgg19_load_image.setText(_translate("Form", "Load Image"))
        self.vgg19_show_ar_images.setText(_translate("Form", "5.1 Show Augmented \n""Images"))
        self.show_model_structure.setText(_translate("Form", "5.2  Show Model Structure"))
        self.show_acc_and_loss.setText(_translate("Form", "5.3 Show Acc and Loss"))
        self.inference.setText(_translate("Form", "5.4 Inference"))
        self.label.setText(_translate("Form", "Predict = "))
        self.groupBox_7.setTitle(_translate("Form", "4. SIFT"))
        self.load_image1.setText(_translate("Form", "Load Image1"))
        self.load_image2.setText(_translate("Form", "Load Image2"))
        self.keypoints.setText(_translate("Form", "4.1 Keypoints"))
        self.matched_keypoints.setText(_translate("Form", "4.2 MatchedKeypoints"))

