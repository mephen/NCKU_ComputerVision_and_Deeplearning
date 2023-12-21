import sys
import random
import enum
from typing import List, Tuple
import psutil
import cv2
import os
import numpy as np
import torch.nn as nn
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets


import matplotlib
import matplotlib.pyplot as plt
from PIL import Image 
import torch
import torchvision
import torchvision.transforms as transforms
import torchsummary

img = None
video = None

def load_img():
    img_path, _ = QFileDialog.getOpenFileName(None, "Open File", r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw2\Dataset_OpenCvDl_Hw2", "Image Files (*.jpg *.jpeg *.png *.bmp)", None) 
    global img
    img = cv2.imread(img_path)

def load_video():
    video_path, _ = QFileDialog.getOpenFileName(None, 'Open File', r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw2\Dataset_CvDl_Hw2", "Video Files (*.mp4 *.avi);;All Files (*)", None)
    global video
    video = cv2.VideoCapture(video_path)

def resize_show(string, img, width, height):
    shape_ratio = min(600/img.shape[1], 400/img.shape[0])
    resized_img = cv2.resize(img, None, fx=shape_ratio, fy=shape_ratio)
    cv2.imshow(string,resized_img)
    cv2.moveWindow(string, width, height)#通常是 (50,50)，若要左右比較可改成 (450, 50)
    if cv2.waitKey(1000) != -1:
        cv2.destroyAllWindows()

#Background_Subtraction
class Assign_1:
    def Background_Subtraction(self):
        # Create a background subtractor
        subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

        global video
        while True:
            ret, frame = video.read()

            if not ret:
                break

            # Blur the frame
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # Apply background subtractor to get the mask
            mask = subtractor.apply(blurred_frame)

            # Create a side-by-side display of the original frame, foreground mask, and backgound mask
            display_frame = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.bitwise_and(frame, frame, mask=mask)))

            # Display the result
            cv2.imshow("Result", display_frame)

            # Break the loop if 'Esc' key is pressed
            if cv2.waitKey(30) == 27:
                break

        video.release()
        cv2.destroyAllWindows()
    

#Optical flow
class Assign_2:
    def __init__(self):
        self.old_frame = None
        self.old_gray  = None
        self.nose_point = None
        self.canvas = None

    def Preprocessing(self):
        video = cv2.VideoCapture(r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw2\Dataset_CvDl_Hw2\Q2\optical_flow.mp4")
        # Read the first frame
        ret, self.old_frame = video.read()
        if ret:
            # Convert the frame to grayscale
            self.old_gray  = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)

            # Use cv2.goodFeaturesToTrack to detect the point at the bottom of the doll's nose
            self.nose_point = cv2.goodFeaturesToTrack(self.old_gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

            if self.nose_point is not None:
                # Extract the coordinates of the detected point
                x, y = self.nose_point[0].ravel()

                # Draw a red cross mark at the detected point
                cv2.line(self.old_frame, (int(x) - 10, int(y)), (int(x) + 10, int(y)), (0, 0, 255), 4)
                cv2.line(self.old_frame, (int(x), int(y) - 10), (int(x), int(y) + 10), (0, 0, 255), 4)
                
                # Display the frame using cv2.imshow
                resize_show('Frame with Nose Point', self.old_frame, 50, 50)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Video_tracking(self):
        video = cv2.VideoCapture(r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw2\Dataset_CvDl_Hw2\Q2\optical_flow.mp4")
        # set min size of tracked object, e.g. 15x15px
        parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.old_frame)
        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        while True:
            # get next frame
            ok, frame = video.read()
            if not ok:
                print("[INFO] end of file reached")
                break
            
            # prepare grayscale image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # update object corners by comparing with found self.nose_point in initial frame
            update_nose_point, status, errors = cv2.calcOpticalFlowPyrLK(
                self.old_gray, 
                frame_gray, 
                self.nose_point, 
                None,
                **parameter_lucas_kanade,
            )

            # only update self.nose_point if algorithm successfully tracked
            new_nose_point = update_nose_point[status == 1]
            # to calculate directional flow we need to compare with previous position
            old_nose_point = self.nose_point[status == 1]


            for i, (new, old) in enumerate(zip(new_nose_point, old_nose_point)):
                a, b = new.ravel()
                c, d = old.ravel()

                # draw line between old and new corner point with random colour
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                # draw circle around new position
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            result = cv2.add(frame, mask)

            cv2.imshow('Optical Flow', result)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # overwrite initial frame with current before restarting the loop
            self.old_gray = frame_gray.copy()
            # update to new edges before restarting the loop
            self.nose_point = new_nose_point.reshape(-1, 1, 2)
        
        video.release()
        cv2.destroyAllWindows()

#PCA – Dimension Reduction
class Assign_3:
    def pca(self):
        img = cv2.imread(r"D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw2\Dataset_CvDl_Hw2\Q3\logo.jpg")
        gray_image  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize gray scale image
        normalized_gray_image = np.float32(gray_image) / 255.0
        # flat_image = normalized_gray_image.flatten()
        width, high = normalized_gray_image.shape
        min_components_number = None

        for n in range(1, 350):
            # Apply PCA
            pca = PCA(n_components=n)
            # reduced_image = pca.fit_transform(flat_image.reshape(1, -1))
            reduced_image = pca.fit_transform(normalized_gray_image)
            reconstructed_image = pca.inverse_transform(reduced_image)

            # Calculate Mean Squared Error
            mse = 0.0
            for i in range(width):
                for j in range(high):
                    mse += (normalized_gray_image[i][j]-reconstructed_image[i][j])**2
            # mse = mse**0.5
            # mse = mse / (width*high)

            # Check if the error is less than or equal to the threshold
            if mse <= 3.0:
                print(mse)
                print(f"n value = {n}")
                min_components_number = n
                break

        # Reconstruct the image using the selected components
        pca = PCA(n_components=min_components_number)
        reduced_image = pca.fit_transform(normalized_gray_image)
        reconstructed_image = pca.inverse_transform(reduced_image)

        # Reshape the reconstructed image
        reconstructed_image = reconstructed_image.reshape(gray_image.shape)

        # Plot the original and reconstructed images
        resize_show("Original Image", img, 50, 50)
        resize_show("gray Image", gray_image, 450, 50)
        resize_show(f"Reconstructed Image(n={min_components_number})", reconstructed_image, 850, 50)

class Assign_4:
    def show_model(self):
        model = torchvision.models.vgg19_bn()
        torchsummary.summary(model, (3, 224, 224))
    
    def show_acc_and_loss(self):
        img = cv2.imread("./assign_4_cureve.png")
        if img is not None:
            cv2.imshow("accuracy and loss", img)
            cv2.moveWindow("accuracy and loss", 50 , 50)#通常是 (50,50)，若要左右比較可改成 (450, 50)
            if cv2.waitKey(1000) != -1:
                cv2.destroyAllWindows()
        else:
            print("Error: Failed to read the image.")
    
    def predict(self):
        # 定義圖片轉換
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        
        img = Image.open("./test.png")
        test_img = transform(img).unsqueeze(0)
        model = torch.load('./assign_4_model.pth')
        model.eval()
        # 進行模型推論
        with torch.no_grad():
            output = model(test_img)

        # 獲取預測標籤
        _, predicted_label = torch.max(output, 1)
        _ , index = torch.max(output, 1) # 通过argmax方法，得到概率最大的处所对应的索引
        # 模型預測的概率分佈
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        class_probabilities = np.sum(probabilities.numpy().reshape(1, -1), axis=1)

        img_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

        x_labels = img_classes
        plt.bar(np.array(img_classes), class_probabilities)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Probability of each class')
        plt.xticks(rotation=45, ha='right') #調整標籤顯示方式，讓x軸上的標籤更容易閱讀。
        plt.tight_layout()

        # 顯示直方圖
        plt.show()



# 定義推論數據集類別
class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class Assign_5:
    def __init__(self):
        self.inference_dataset = None

    def load_img(self):
        # 設定推論數據集的路徑
        inference_path = r'D:\OneDrive - gs.ncku.edu.tw\oneDrive_NCKU_master\1st_semester\ComputerVision_and_Deeplearning\hw\hw2\Dataset_CvDl_Hw2\Q5_dataset\inference_dataset'

        # 定義轉換
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        # 創建推論數據集
        self.inference_dataset = InferenceDataset(inference_path, transform=None)

    def show_imgs(self):
        # 顯示一個來自每個類別的圖像
        class_labels = self.inference_dataset.dataset.classes
        fig, axs = plt.subplots(1, len(class_labels))

        for i, label in enumerate(class_labels):
            # 找到該類別的所有圖像索引
            indices = [index for index, label in enumerate(self.inference_dataset.dataset.targets) if label == i]
            
            # 從該類別的所有圖像中隨機選擇一個
            random_index = random.choice(indices)
            
            # 獲取圖像路徑
            img_path = self.inference_dataset.dataset.samples[random_index][0]
            img = plt.imread(img_path)

            # 顯示圖像
            axs[i].imshow(img, extent=[0, 224, 0, 224])
            axs[i].set_title(label)
            axs[i].axis('off')

        plt.show()

    def show_model(self):
        model = torchvision.models.resnet50(num_classes=2)
        model.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        torchsummary.summary(model, (3, 224, 224))
    
    def show_comparison(self):
        pass