import enum
from typing import List, Tuple
import psutil
import cv2
import os
import numpy as np
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image 
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary


#a bunch of images in the folder
folder_list = []
img_l = None
img_r = None


class LoadImage:
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None , "choose folder path",r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", options = QFileDialog.ShowDirsOnly)
        #print(f"folder path: {folder_path}")
        #filter image from folder
        for name in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, name)) and name.endswith((".bmp", ".jpg", ".jpeg", ".png")):
                folder_list.append(name)
        #先排序檔案名稱，再在檔名前面加入檔案路徑
        folder_list.sort(key=self.custom_sort_key)
        for index, img_name in enumerate(folder_list):
            img_path = os.path.join(folder_path, img_name)
            img_path = img_path.replace("\\", "/") #將 \ 替換成 /
            folder_list[index] = img_path
        #print(folder_list)
        #將這些img_path 讀取進 folder_list 中。
        for index, img_path in enumerate(folder_list):
            img = cv2.imread(img_path)
            #不要在還沒執行圖像處理前就對圖像大小作改動，否則後續可能出問題。(e.g. 找不到角點)
            folder_list[index] = img
        #print(folder_list)

    def load_image_L(self):
        img_l_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None)
        global img_l 
        img_l = cv2.imread(img_l_path)
        #resize_show("img_l", img_l, 50, 50)

    def load_image_R(self):
        img_r_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None)
        global img_r
        img_r = cv2.imread(img_r_path)
        #resize_show("img_r", img_r, 50, 50)

    # 自定义排序函数，将字符串中的数字提取并转换为整数
    def custom_sort_key(self, item):
        parts = item.split(".")
        if (len(parts) == 2) and (parts[0].isdigit()):
            return int(parts[0])
        return 0

def resize_show(string, img, width, height):
    shape_ratio = min(800/img.shape[1], 600/img.shape[0])
    resized_img = cv2.resize(img, None, fx=shape_ratio, fy=shape_ratio)
    cv2.imshow(string,resized_img)
    cv2.moveWindow(string, width, height)#通常是 (50,50)，若要左右比較可改成 (850, 50)
    if cv2.waitKey(1000) != -1:
        cv2.destroyAllWindows()

class CameraCalibration:    
    def __init__(self):
        self.retval = None
        self.intrinsic_mat = None
        self.distort_mat= None
        self.rotation_vecs = None
        self.traslation_vecs= None
        self.extrinsic_mat = []
        self.corners = []
    
    def calibrate(self):
        self.corners.clear()
        for _, img in enumerate(folder_list):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.found, corners = cv2.findChessboardCorners(gray_img, (11, 8), None) #11*8 是內角點(行x列)
            if self.found:
                corners = cv2.cornerSubPix(gray_img, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            self.corners.append(corners)
        #print(len(self.corners))

        #cal intrinsic
        objectPoints = []  # 用于存储物体点的列表
        imagePoints = []   # 用于存储图像点的列表
        image_size = (folder_list[0].shape[1], folder_list[0].shape[0])  # 存储图像尺寸的元组
        # 生成标定板上物体点的三维坐标
        objp = np.zeros((8 * 11, 3), np.float32) #創造 8*11 的 (0, 0, 0)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) #將objp的所有行(:)和前两列(:2)的數組做操作：先生成多维网格数据，第一維0-10，第二維0-7，取轉置，然後再轉換成一個兩列的數組。
        for index, _ in enumerate(folder_list):
            objectPoints.append(objp)  # 生成objp并添加到objectPoints
            imagePoints.append(self.corners[index])  # 添加角點到imagePoints
        # 此时 objectPoints，imagePoints 和 image_size 都已经生成
        # print("Object Points:", objectPoints)
        # print("Image Points:", imagePoints)
        # print("Image Size:", image_size)
        self.retval, self.intrinsic_mat, self.distort_mat, self.rotation_vecs, self.traslation_vecs = cv2.calibrateCamera(objectPoints, imagePoints, image_size, None, None)
        
        #cal all the extrinsic
        if self.retval:
            self.extrinsic_mat.clear()
            for index in range(len(folder_list)):
                rotation_matrix, _ = cv2.Rodrigues(self.rotation_vecs[index])#將旋轉信號->旋轉矩陣
                self.extrinsic_mat.append(np.hstack((rotation_matrix, self.traslation_vecs[index])))

    def find_corners_and_show_pictures(self):
        self.calibrate()
        for index, img in enumerate(folder_list):
            print(f"type of corners: {type(self.corners[index])}")
            cv2.drawChessboardCorners(folder_list[index], (11, 8), self.corners[index], self.found)
            shape_ratio = min(800/img.shape[1], 600/img.shape[0])
            resized_img = cv2.resize(img, None, fx=shape_ratio, fy=shape_ratio)
            cv2.imshow('image', resized_img)
            cv2.moveWindow('image', 50, 50)
            # 等待 x 毫秒后进入下一次循环
            if cv2.waitKey(500) != -1:
                cv2.destroyAllWindows()


    #将相机坐标系中的点投影到图像平面上的像素坐标
    def find_intrinsic_and_show(self):
        self.calibrate()
        # objectPoints = []  # 用于存储物体点的列表
        # imagePoints = []   # 用于存储图像点的列表
        # image_size = (folder_list[0].shape[1], folder_list[0].shape[0])  # 存储图像尺寸的元组
        # # 生成标定板上物体点的三维坐标
        # objp = np.zeros((8 * 11, 3), np.float32) #創造 8*11 的 (0, 0, 0)
        # objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) #將objp的所有行(:)和前两列(:2)的數組做操作：先生成多维网格数据，第一維0-10，第二維0-7，取轉置，然後再轉換成一個兩列的數組。
        # for index, _ in enumerate(folder_list):
        #     objectPoints.append(objp)  # 生成objp并添加到objectPoints
        #     imagePoints.append(self.corners[index])  # 添加角點到imagePoints
        # # 此时 objectPoints，imagePoints 和 image_size 都已经生成
        # # print("Object Points:", objectPoints)
        # # print("Image Points:", imagePoints)
        # # print("Image Size:", image_size)
        # self.retval, self.intrinsic_mat, self.distort_mat, self.rotation_vecs, self.traslation_vecs = cv2.calibrateCamera(objectPoints, imagePoints, image_size, None, None)
        print("Intrinsic matrix:")
        print(self.intrinsic_mat, "\n")


    #要先找到 intrinsic matrix
    def find_extrinsic_and_show(self, index):
        self.calibrate()
        print(f"extrinsic_mat{index-1}", self.extrinsic_mat[index-1], "\n")


    
    def find_distortion_and_show(self):
        if self.retval:
            print("Distortion matrix:")
            print(self.distort_mat, "\n")
        else:
            print("should run Find intrinsic first.")


    def show_undistorted(self):
        if self.retval:
            distorted_imgs = [] #original image
            undistorted_imgs = []
            for index in range(len(folder_list)):
                img_size = (folder_list[index].shape[1], folder_list[index].shape[0])#width, high
                #計算優化的相機矩陣，然露糾正圖像畸變。
                #1是縮放因子代表不縮放。roi：裁剪图像的区域，包含(x,y,high, width)。
                #None表示输出图像的相机矩阵，使用默認值。
                camera_mat, roi = cv2.getOptimalNewCameraMatrix(self.intrinsic_mat, self.distort_mat, img_size, 1, img_size)
                undistort_img = cv2.undistort(folder_list[index], self.intrinsic_mat, self.distort_mat, None, camera_mat)
                x_axis, y_axis, width, high = roi
                #print(x_axis, y_axis)
                #割掉不要的部分，只留下 y~y+h列 和 x~x+w行
                undistort_img = undistort_img[y_axis:y_axis+high, x_axis:x_axis+width]
                distorted_imgs.append(cv2.resize(folder_list[index], (600, 600)))
                undistorted_imgs.append(cv2.resize(undistort_img, (600, 600)))
            
            for index in range(len(distorted_imgs)):
                cv2.imshow("distorted", distorted_imgs[index])
                cv2.imshow("Undistorted", undistorted_imgs[index])
                cv2.moveWindow('distorted', 50, 50)
                cv2.moveWindow('Undistorted', 850, 50)
                if cv2.waitKey(500) != -1:
                    cv2.destroyAllWindows()
        else:
            print("should run Find intrinsic first.")

def check_upper(words):
    for index in range(len(words)):
        if not(words[index].isupper()):
            print("words should be upper")
            return False
    return True


class AugmentedReality:
    def __init__(self):
        self.cameraCalibration = CameraCalibration()
        self.word_button_right_position = [[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]] #order:1,2,3,4,5,6

    def show_words_horizontal(self, words):
        if(len(words) > 6) or not(check_upper(words)):
            print("wrong words parameter")
            return
        
        alphabet_lib_on_board = r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1\Q2_Image\Q2_lib\alphabet_lib_onboard.txt"
        fs = cv2.FileStorage(alphabet_lib_on_board, cv2.FileStorage_READ)
        
        #get mat of the input char，由多個 stroke 構成
        #stroke 代表字母的某一劃
        char_stroke_matrix = []
        for char in words:
            char_stroke_matrix.append(fs.getNode(char).mat())

        #取得 char 在圖像上的位置(img_char_matrix)
        img_char_matrix = []
        #char_mat 代表字母
        for char_idx, char_mat in enumerate(char_stroke_matrix):
            #print(char_mat.shape[0]) #以A為例，shape會返回(5,2,3)，代表是三維矩陣。(A有5個筆畫，每個筆畫是2*3的矩陣)
            #計算每個筆畫在圖像上的起始和結束位置
            for stroke_idx in range(char_mat.shape[0]):
                stroke_src = char_stroke_matrix[char_idx][stroke_idx][0] + self.word_button_right_position[char_idx]
                stroke_end = char_stroke_matrix[char_idx][stroke_idx][1] + self.word_button_right_position[char_idx]
                img_char_matrix.append( (stroke_src, stroke_end) )

        #do cameraCalibration
        self.cameraCalibration.calibrate()
        objpts = np.zeros((8*11, 3), np.float32)
        objpts[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        #計算圖片上的字的多個 stroke 的投射位置。
        project_points = []
        for index in range(len(folder_list)):
            _, intrinsic, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera([objpts], [self.cameraCalibration.corners[index]], (2048, 2048), None, None)
            # 投射 img_char_matrix，然後劃線
            imgpts, _ = cv2.projectPoints(
                np.float32(img_char_matrix).reshape(-1, 3),
                np.array(rotation_vectors),
                np.array(translation_vectors).reshape(3, 1),
                intrinsic,
                distortion
            )
            project_points.append(imgpts)
        #print(project_points)
        char_strokes_on_image = np.array(project_points).reshape(len(folder_list), len(img_char_matrix), 2, 2)
        char_on_imgs = []
        for index, img in enumerate(folder_list):
            output_img = img.copy()
            for char_stroke in char_strokes_on_image[index]:
                stroke_start = (int(char_stroke[0][0]), int(char_stroke[0][1]))
                stroke_end = (int(char_stroke[1][0]), int(char_stroke[1][1]))
                # print(stroke_start)
                
                #劃紅線
                output_img = cv2.line(output_img, stroke_start, stroke_end, (0, 0, 255), 10)
            output_img = cv2.resize(output_img, (512, 512))
            char_on_imgs.append(output_img)
        
        for img in char_on_imgs:
            resize_show("char_on_image", img, 50, 50)


    def show_words_vertical(self, words):
        if(len(words) > 6) or not(check_upper(words)):
            print("wrong words parameter")
            return
        
        alphabet_lib_on_board = r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1\Q2_Image\Q2_lib\alphabet_lib_vertical.txt"
        fs = cv2.FileStorage(alphabet_lib_on_board, cv2.FileStorage_READ)
        
        #get mat of the input char，由多個 stroke 構成
        #stroke 代表字母的某一劃
        char_stroke_matrix = []
        for char in words:
            char_stroke_matrix.append(fs.getNode(char).mat())

        #取得 char 在圖像上的位置(img_char_matrix)
        img_char_matrix = []
        #char_mat 代表字母
        for char_idx, char_mat in enumerate(char_stroke_matrix):
            #print(char_mat.shape[0]) #以A為例，shape會返回(5,2,3)，代表是三維矩陣。(A有5個筆畫，每個筆畫是2*3的矩陣)
            #計算每個筆畫在圖像上的起始和結束位置
            for stroke_idx in range(char_mat.shape[0]):
                stroke_src = char_stroke_matrix[char_idx][stroke_idx][0] + self.word_button_right_position[char_idx]
                stroke_end = char_stroke_matrix[char_idx][stroke_idx][1] + self.word_button_right_position[char_idx]
                img_char_matrix.append( (stroke_src, stroke_end) )

        #do cameraCalibration
        self.cameraCalibration.calibrate()
        objpts = np.zeros((8*11, 3), np.float32)
        objpts[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        #計算圖片上的字的多個 stroke 的投射位置。
        project_points = []
        for index in range(len(folder_list)):
            _, intrinsic, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera([objpts], [self.cameraCalibration.corners[index]], (2048, 2048), None, None)
            # 投射 img_char_matrix，然後劃線
            imgpts, _ = cv2.projectPoints(
                np.float32(img_char_matrix).reshape(-1, 3),
                np.array(rotation_vectors),
                np.array(translation_vectors).reshape(3, 1),
                intrinsic,
                distortion
            )
            project_points.append(imgpts)
        #print(project_points)
        char_strokes_on_image = np.array(project_points).reshape(len(folder_list), len(img_char_matrix), 2, 2)
        char_on_imgs = []
        for index, img in enumerate(folder_list):
            output_img = img.copy()
            for char_stroke in char_strokes_on_image[index]:
                stroke_start = (int(char_stroke[0][0]), int(char_stroke[0][1]))
                stroke_end = (int(char_stroke[1][0]), int(char_stroke[1][1]))
                # print(stroke_start)
                
                #劃紅線
                output_img = cv2.line(output_img, stroke_start, stroke_end, (0, 0, 255), 10)
            output_img = cv2.resize(output_img, (512, 512))
            char_on_imgs.append(output_img)
        
        for img in char_on_imgs:
            resize_show("char_on_image", img, 50, 50)

#視差圖
class StereoDisparityMap:
    def stereo_disparity_map(self):
        if(img_l is None) or (img_r is None):
            print("load image L & R")
            return
        stereo = cv2.StereoBM_create(256, 25)
        #創建視差圖(min_disp 預設為 0)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(gray_l, gray_r)
        #規範化視差圖以进行可视化。
        disparity = disparity/16
        #將視差值映射到0-255，以便更容易可视化和分析视差图。normalize(src, dst, alpha, beta, norm_type, dtype, mask)
        disparity = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #print(disparity)
        resize_show('disparity', disparity, 50, 50)
        if cv2.waitKey(500) != -1:
            cv2.destroyAllWindows()

        #注意：imgL 和 imgR 不太能 resize！否則計算 mouse_disparity 的時候會出現誤差，無法正確圈出滑鼠位置。
        #check the disparity value(click response)
        cv2.namedWindow("imgL", cv2.WINDOW_NORMAL)
        cv2.imshow("imgL",img_l)
        cv2.moveWindow("imgL", 50, 450)

        def mouse_clicked(event, x, y, ignore1, ignore2):
            if event is cv2.EVENT_LBUTTONDOWN:
                imgR_copy = img_r.copy()
                #取得滑鼠點擊在視差圖中點擊位置的視差值：左右立体图像之间对应像素之间的"水平"距离差异。
                #要先將滑鼠坐標系轉換到圖象坐標系 (e.g. 先橫再直(3,4) -> 先直再橫(4,3))，再轉換成整數
                #disparity at (x,y) = clicked_disparity
                mouse_disparity = int(disparity[y][x])
                print(f'({x},{y}), dis:{mouse_disparity}')
                if mouse_disparity == 0: #失敗案例
                    print('fail, ignore.')
                else:
                    cv2.namedWindow("imgR", cv2.WINDOW_NORMAL)
                    #x-clicked_disparity 是要讓右圖扣掉視差，找到在左视图中的滑鼠位置。
                    circle_imgR = cv2.circle(imgR_copy, (x-mouse_disparity, y), 5, (0, 255, 0), 10)
                    cv2.moveWindow("imgR", 950, 450)
                    
                    shape_ratio = min(800/imgR_copy.shape[1], 600/imgR_copy.shape[0])
                    resized_img = cv2.resize(imgR_copy, None, fx=shape_ratio, fy=shape_ratio)
                    cv2.imshow("imgR",resized_img)
                    if cv2.waitKey(500) != -1:
                        cv2.destroyAllWindows()
                    #cv2.resizeWindow("imgR", int(imgR_copy.shape[1]*shape_ratio), int(imgR_copy.shape[0]*shape_ratio))
                    cv2.imshow("imgR", imgR_copy)
        
        cv2.setMouseCallback("imgL", mouse_clicked)


#跑的時候不要動它，不然容易當機。
class SIFT:
    def __init__(self):
        self.img1 = None
        self.img2 = None
    
    def load_image1(self):
        img1_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None) 
        self.img1 = cv2.imread(img1_path)
        # resize_show("img1", self.img1, 50, 50)
    
    def load_image2(self):
        img2_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None) 
        self.img2 = cv2.imread(img2_path)
        # resize_show("img2", self.img2, 50, 50)

    def key_points(self):
        sift = cv2.SIFT.create() 
        #使用SIFT检测器检测关键点，然後計算description
        gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_img1, None)
        #建立灰階圖，方便绘制关键点
        keypoints_img = cv2.drawKeypoints(gray_img1, keypoints, None, (0, 255, 0))
        resize_show('SIFT Keypoints', keypoints_img, 50, 50)
    
    def matched_keypoints(self):
        #計算keypoints 和 descriptors，方式同 key_points()。
        #變灰階處理比較快
        sift = cv2.SIFT.create() 
        gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        keypoints_img1, descriptors_img1 = sift.detectAndCompute(gray_img1, None)
        keypoints_img2, descriptors_img2 = sift.detectAndCompute(gray_img2, None)

        bfMatch = cv2.BFMatcher()
        matches = bfMatch.knnMatch(descriptors_img1, descriptors_img2, k=2)

        good_matches = [] 
        for m,n in matches:
            if m.distance < (0.75 * n.distance):
                good_matches.append([m])
        
        #注意：matches1to2 要放 good_matches；matchesMask 不要放。
        match_image = cv2.drawMatchesKnn(gray_img1, keypoints_img1, gray_img2, keypoints_img2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        resize_show("Matching image", match_image, 50, 50)


class VGG19:
    def __init__(self):
        self.img_path = None
        self.img = None
    
    def load_img(self, gui_object):
        img_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None) 
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        # resize_show("img", self.img, 50, 50)
        # return self.img
        pixmap = QPixmap(self.img_path).scaled(128, 128)  # 載入並調整大小為128x128
        gui_object.scene.clear()
        gui_object.scene.addPixmap(pixmap)
    
    def show_augmented_images(self):
        #get 9 img_path in /Q5_image/Q5_1/ 
        folder_path = r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1\Q5_image\Q5_1"
        folder = []
        for name in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, name)) and name.endswith((".bmp", ".jpg", ".jpeg", ".png")):
                folder.append(name)
        for index, img_name in enumerate(folder):
            img_path = os.path.join(folder_path, img_name)
            img_path = img_path.replace("\\", "/") #將 \ 替換成 /
            folder[index] = img_path
        # print(folder)
        
        #set 3 type of data augmentation
        RandomRotation = transforms.Compose([
        transforms.Resize((600,600)),
        transforms.RandomRotation(180), #隨機旋轉，最多180度
        ])

        RandomHorizontalFlip = transforms.Compose([
        transforms.Resize((600,600)),
        transforms.RandomHorizontalFlip(p=1)#p=1 代表100%要翻轉
        ])

        RandomVertocalFlip = transforms.Compose([
        transforms.Resize((600,600)),
        transforms.RandomVerticalFlip(p=1)#p=1 代表100%要翻轉
        ])

        #load 9 imgs in /Q5_image/Q5_1/ 
        img_list = []
        augmented_img_list = []
        for index, img_path in enumerate(folder):
            im = Image.open(img_path)
            print(f"{os.path.basename(img_path)} loaded")
            img_list.append(im)
            # #faster than im.show()
            # show_im = cv2.imread(img_path)
            # # resize_show("original", show_im, 50, 50)
            # cv2.imshow("original", show_im)

        #dp transformation
        for index in range(len(img_list)):
            if index < 3:
                augmented_img_list.append(RandomRotation(img_list[index]))
            elif index >=3 and index < 6:
                augmented_img_list.append(RandomHorizontalFlip(img_list[index]))
            else:
                augmented_img_list.append(RandomVertocalFlip(img_list[index]))

        # 创建一个9x9的图形布局，axes: 多個子圖
        ignore_figure, axes = plt.subplots(3, 3, figsize=(8, 8))
        plt.ion() #使用 plt.ion() 和 plt.ioff() 可以避免跑出錯誤訊息：QCoreApplication::exec: The event loop is already running。(因為它能夠讓 Matplotlib 更好地與圖形介面的事件循環進行整合。)
        # 使用循环将每张图像放到对应的子图中
        for i, ax in enumerate(axes.ravel()):
            # 将PIL图像转换为OpenCV格式
            show_im = cv2.cvtColor(np.array(augmented_img_list[i]), cv2.COLOR_RGB2BGR)
            #faster than im.show()
            ax.imshow(show_im[:,:,[2,1,0]]) # 调整一下第三维的通道即可，避免 plt 顯示 opencv 的圖像出現色差(因为前两维是宽和高，不用调整)
            ax.set_title(f'{os.path.basename(folder[i])}')
        # 显示图像
        plt.tight_layout()
        plt.show()
        plt.ioff()
    
    def show_moddel_structure(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vgg19_bn = torchvision.models.vgg19(num_classes = 10).to(device)   #載入VGG19 model and pretrained parameter
        print(vgg19_bn)
        summary(vgg19_bn,(3,32,32))

        # # model by my self in hw1_train.py
        # model = torch.load('model.pth',device)
        # #32*32 是Q5_1中的圖像大小
        # summary(model, (3,32,32))#batch_size不寫，device參數則是會自己去抓
        
    def show_acc_and_loss(self):
        # img_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_vscode\OpenCv_DeepLearning", "Image Files (*.jpg *.jpeg *.png *.bmp)", None)
        cuve_img = cv2.imread(r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_vscode\OpenCv_DeepLearning\cureve.png")
        resize_show("curve image", cuve_img, 50, 50)

    def show_inference(self):
        if(self.img is None):
            print("Should load image first")
            return
        
        img_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #注意！這邊使用的 image_transform_for_test 跟 hw1_train.py 的 test_dataset 使用相同的 transform 可以得到比較好的 inference。
        image_transform_for_test = transforms.Compose([
                            transforms.RandomCrop(size=32, padding=4),  # 裁減
                            transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        model = torch.load('model.pth',device)  # 使用 map_location 將模型移到相同的設備)

        #進入測試模式
        model.eval()
        plt.ion()

        plt.clf()  # 用來清除上一次產生的圖形

        #resize_show("image" , self.img, 50, 50)
        #test_img = cv2.imread(r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_vscode\OpenCv_DeepLearning\cureve.png")
        test_img = Image.open(self.img_path)
        #將側視圖片做調整、如果只有一張圖的話，需要用 unsqueeze_(0) 將三維的陣列擴增為四維。
        test_img_tensor = image_transform_for_test(test_img).unsqueeze(0).to(device) # 將input移到相同的設備
        output = model(test_img_tensor)
        _ , index = torch.max(output, 1) # 通过argmax方法，得到概率最大的处所对应的索引
        # 模型預測的概率分佈
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        x_labels = img_classes
        plt.bar(np.array(img_classes), probabilities.cpu().detach().numpy()) # 使用 .cpu() 將數據移到 CPU，才能夠進行 numpy() 操作
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Probability of each class')
        plt.xticks(rotation=45, ha='right') #調整標籤顯示方式，讓x軸上的標籤更容易閱讀。
        plt.tight_layout()

        # 顯示直方圖
        plt.show()

