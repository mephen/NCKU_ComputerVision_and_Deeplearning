# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Hw2_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF
from PyQt5 import QtCore, QtGui, QtWidgets
from hw2_processing import Assign_1, load_video, load_img, Assign_2, Assign_3, Assign_4, Assign_5

assign_1 = Assign_1()
assign_2 = Assign_2()
assign_3 = Assign_3()
assign_4 = Assign_4()
assign_5 = Assign_5()

from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF
import sys

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1099, 699)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(110, 40, 221, 91))
        self.groupBox.setObjectName("groupBox")
        self.Background_substraction = QtWidgets.QPushButton(self.groupBox)
        self.Background_substraction.setGeometry(QtCore.QRect(30, 40, 161, 23))
        self.Background_substraction.setObjectName("Background_substraction")
        self.Background_substraction.clicked.connect(assign_1.Background_Subtraction)

        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(360, 40, 681, 261))
        self.groupBox_4.setObjectName("groupBox_4")
        self.show_model_structure = QtWidgets.QPushButton(self.groupBox_4)
        self.show_model_structure.setGeometry(QtCore.QRect(20, 30, 171, 23))
        self.show_model_structure.setObjectName("show_model_structure")
        self.show_model_structure.clicked.connect(assign_4.show_model)

        self.Show_accuracy_and_loss = QtWidgets.QPushButton(self.groupBox_4)
        self.Show_accuracy_and_loss.setGeometry(QtCore.QRect(20, 80, 171, 23))
        self.Show_accuracy_and_loss.setObjectName("Show_accuracy_and_loss")
        self.Show_accuracy_and_loss.clicked.connect(assign_4.show_acc_and_loss)

        self.Predict = QtWidgets.QPushButton(self.groupBox_4)
        self.Predict.setGeometry(QtCore.QRect(20, 130, 171, 23))
        self.Predict.setObjectName("Predict")
        self.Predict.clicked.connect(assign_4.predict)
        

        self.Reset = QtWidgets.QPushButton(self.groupBox_4)
        self.Reset.setGeometry(QtCore.QRect(20, 180, 171, 23))
        self.Reset.setObjectName("Reset")


        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_4)
        self.graphicsView.setGeometry(QtCore.QRect(280, 30, 351, 192))
        self.graphicsView.setObjectName("graphicsView")


        self.Load_Image = QtWidgets.QPushButton(Form)
        self.Load_Image.setGeometry(QtCore.QRect(20, 120, 75, 23))
        self.Load_Image.setObjectName("Load_Image")
        self.Load_Image.clicked.connect(load_img)

        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(110, 180, 221, 111))
        self.groupBox_2.setObjectName("groupBox_2")
        self.Preprocessing = QtWidgets.QPushButton(self.groupBox_2)
        self.Preprocessing.setGeometry(QtCore.QRect(20, 30, 171, 23))
        self.Preprocessing.setObjectName("Preprocessing")
        self.Preprocessing.clicked.connect(assign_2.Preprocessing)

        self.video_tracking = QtWidgets.QPushButton(self.groupBox_2)
        self.video_tracking.setGeometry(QtCore.QRect(20, 70, 171, 23))
        self.video_tracking.setObjectName("video_tracking")
        self.video_tracking.clicked.connect(assign_2.Video_tracking)

        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setGeometry(QtCore.QRect(110, 350, 221, 81))
        self.groupBox_3.setObjectName("groupBox_3")
        self.Dimension_reduction = QtWidgets.QPushButton(self.groupBox_3)
        self.Dimension_reduction.setGeometry(QtCore.QRect(30, 30, 171, 23))
        self.Dimension_reduction.setObjectName("Dimension_reduction")
        self.Dimension_reduction.clicked.connect(assign_3.pca)

        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setGeometry(QtCore.QRect(360, 350, 681, 261))
        self.groupBox_5.setObjectName("groupBox_5")
        self.Load_image_assign5 = QtWidgets.QPushButton(self.groupBox_5)
        self.Load_image_assign5.setGeometry(QtCore.QRect(20, 30, 171, 23))
        self.Load_image_assign5.setObjectName("Load_image_assign5")
        self.Load_image_assign5.clicked.connect(assign_5.load_img)

        self.Show_images = QtWidgets.QPushButton(self.groupBox_5)
        self.Show_images.setGeometry(QtCore.QRect(20, 80, 171, 23))
        self.Show_images.setObjectName("Show_images")
        self.Show_images.clicked.connect(assign_5.show_imgs)

        self.show_model_structure_assign5 = QtWidgets.QPushButton(self.groupBox_5)
        self.show_model_structure_assign5.setGeometry(QtCore.QRect(20, 130, 171, 23))
        self.show_model_structure_assign5.setObjectName("show_model_structure_assign5")
        self.show_model_structure_assign5.clicked.connect(assign_5.show_model)

        self.Show_Comparison = QtWidgets.QPushButton(self.groupBox_5)
        self.Show_Comparison.setGeometry(QtCore.QRect(20, 180, 171, 23))
        self.Show_Comparison.setObjectName("Show_Comparison")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.groupBox_5)
        self.graphicsView_2.setGeometry(QtCore.QRect(280, 30, 351, 192))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.Inference = QtWidgets.QPushButton(self.groupBox_5)
        self.Inference.setGeometry(QtCore.QRect(20, 220, 171, 23))
        self.Inference.setObjectName("Inference")
        self.Load_Video = QtWidgets.QPushButton(Form)
        self.Load_Video.setGeometry(QtCore.QRect(20, 170, 75, 23))
        self.Load_Video.setObjectName("Load_Video")
        self.Load_Video.clicked.connect(load_video)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "1. Background Substraction"))
        self.Background_substraction.setText(_translate("Form", "1. Background Substraction"))
        self.groupBox_4.setTitle(_translate("Form", "4. MNIST Classifier Using VGG19"))
        self.show_model_structure.setText(_translate("Form", "1. Show Model Structure"))
        self.Show_accuracy_and_loss.setText(_translate("Form", "2. Show Accuracy and Loss"))
        self.Predict.setText(_translate("Form", "3. Predict"))
        self.Reset.setText(_translate("Form", "4. Reset"))
        self.Load_Image.setText(_translate("Form", "Load Image"))
        self.groupBox_2.setTitle(_translate("Form", "2. Optical Flow"))
        self.Preprocessing.setText(_translate("Form", "2.1 Preprocessing"))
        self.video_tracking.setText(_translate("Form", "2.2 Video tracking"))
        self.groupBox_3.setTitle(_translate("Form", "3. PCA"))
        self.Dimension_reduction.setText(_translate("Form", "3. Dimension Reduction"))
        self.groupBox_5.setTitle(_translate("Form", "5. ResNet50"))
        self.Load_image_assign5.setText(_translate("Form", "Load Image"))
        self.Show_images.setText(_translate("Form", "5.1 Show Images"))
        self.show_model_structure_assign5.setText(_translate("Form", "5.2 Show Model Structure"))
        self.Show_Comparison.setText(_translate("Form", "5.3 Show Comparison"))
        self.Inference.setText(_translate("Form", "5.4 Inference"))
        self.Load_Video.setText(_translate("Form", "Load Video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
