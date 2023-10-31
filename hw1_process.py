from typing import List, Tuple
import cv2
import os
import numpy as np
# import torch
from PyQt5.QtWidgets import *


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
        #resize_show("img_l", img_l)

    def load_image_R(self):
        img_r_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None)
        global img_r
        img_r = cv2.imread(img_r_path)
        #resize_show("img_r", img_r)

    # 自定义排序函数，将字符串中的数字提取并转换为整数
    def custom_sort_key(self, item):
        parts = item.split(".")
        if (len(parts) == 2) and (parts[0].isdigit()):
            return int(parts[0])
        return 0

def resize_show(string, img):
    shape_ratio = min(800/img.shape[1], 600/img.shape[0])
    resized_img = cv2.resize(img, None, fx=shape_ratio, fy=shape_ratio)
    cv2.imshow(string,resized_img)
    cv2.moveWindow(string, 50, 50)
    if cv2.waitKey(500) != -1:
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
        print(f"extrinsic_mat{index}", self.extrinsic_mat[index], "\n")


    
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
            resize_show("char_on_image", img)


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
            resize_show("char_on_image", img)

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
        resize_show('disparity', disparity)
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
        # resize_show("img1", self.img1)
    
    def load_image2(self):
        img2_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None) 
        self.img2 = cv2.imread(img2_path)
        # resize_show("img2", self.img2)

    def key_points(self):
        sift = cv2.SIFT.create() 
        #使用SIFT检测器检测关键点，然後計算description
        gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_img1, None)
        #建立灰階圖，方便绘制关键点
        keypoints_img = cv2.drawKeypoints(gray_img1, keypoints, None, (0, 255, 0))
        resize_show('SIFT Keypoints', keypoints_img)
    
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
        resize_show("Matching image", match_image)


# class VGG19:
#     def __init__(self):
#         self.img = None
    
#     def load_img(self):
#         img_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw1\Dataset_CvDl_Hw1", "Image Files (*.jpg *.jpeg *.png *.bmp)", None) 
#         self.img = cv2.imread(img_path)
#         resize_show("img", self.img)
#         self.test()
    
#     #
#     def test(self):
#         x = torch.rand(2, 3) 
#         print(x)