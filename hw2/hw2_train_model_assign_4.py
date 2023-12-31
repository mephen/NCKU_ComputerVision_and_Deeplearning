#將使用CIFAR-10資料集載入、訓練VGG19模型，
#並在每個epoch中記錄訓練和驗證的準確度和損失。
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import sys
import matplotlib.pyplot as plt
import numpy as np
#跑出來的圖片 test_accuracy 和 test_loss 很差。
#可能原因：
# 1. 數據轉換原本只有 ToTensor、Normalize 不夠，一定要再至少加入 RandomCrop 和 RandomRotation 之類的才比較夠，增加泛化辨認，能對局部特徵做判斷。
# 2. 原本learning rate = 0.001 ，改成0.0001看看

# 超參數設定
batch_size = 64
epochs = 40
learning_rate = 0.001

# 定義數據轉換
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
])

# 載入訓練數據集
#root：資料集的位置、train=true/false：代表是訓練集/測試集、transform：對資料的預處理、download=True：若root沒有找到資料集，就下載
train_dataset = torchvision.datasets.MNIST(
                    root='./data',
                    train=True,
                    transform=transform,
                    download=True
                )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 載入驗證數據集
test_dataset = torchvision.datasets.MNIST(
                root='./data',
                train=False,
                transform=transform,
                download=True
            )
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定義VGG19模型（包括批次標準化）
model = torchvision.models.vgg19_bn() #沒有預訓練感覺出來成果較差。
# 為了處理灰階影像，調整輸入通道數（原本是 3，改成 1）
model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# 定義損失函數和優化器、訓練中的關鍵
criterion = nn.CrossEntropyLoss() #用於分類任務的損失函數，目標是最小化這個損失函數的值，使模型的預測盡可能接近真實標籤。
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 用於保存訓練和驗證曲線的列表
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def start_training_and_save_model_and_save_picture():
    # 初始化變數以保存最高驗證準確度
    best_accuracy = 0.0
    
    #訓練模型
    #每個epoch中，整個訓練資料集將被遍歷一次，以更新模型的權重。
    for epoch in range(epochs):
        print(epoch)
        model.train() #將模型設置為訓練模式。這一行表示接下來的操作將用於訓練。
        train_loss = 0.0 
        correct = 0
        total = 0 #追蹤總共的樣本數。
        # train_loader_index = 1 #追蹤train每個loop
        # data是包含一個 batch_size 影像資料的張量，target是包含這些影像對應標籤的張量。
        for data, target in train_loader:
            optimizer.zero_grad() #重置優化器的梯度緩衝區，避免受上一次 loop 影響。
            
            outputs = model(data) #產生預測結果
            loss = criterion(outputs, target) #計算模型預測和真實目標之間的損失
            
            loss.backward() #計算損失函數對模型參數的梯度
            optimizer.step() #使用優化器來更新模型的權重
            train_loss += loss.item() 
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # print(f"train_loader_{train_loader_index}", end="; ")
            # train_loader_index += 1
            # sys.stdout.flush()

        train_accuracy = 100.0 * correct / total
        train_loss /= len(train_loader)

        #在驗證集上進行驗證
        model.eval() # 切換到測試模式
        test_loss = 0.0
        correct = 0
        total = 0
        # test_loader_index = 1
        with torch.no_grad(): #停用梯度計算
            for data, target in test_loader:
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # print(f"test_loader_{test_loader_index}", end="; ")
                # test_loader_index += 1
                # sys.stdout.flush()


        test_accuracy = 100.0 * correct / total
        test_loss /= len(test_loader)

        # 將每次 epoch 的訓練和驗證的損失和準確度添加到列表中
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    torch.save(model, './assign_4_model.pth')
    #print('Finished Training')

    #----------------------------------------------------------------
    # 繪製訓練和驗證曲線圖，上下顯示
    plt.figure(figsize=(10, 5))

    # 上側子圖，顯示訓練和驗證損失
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='train_Loss')
    plt.plot(test_losses, label='validation_Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss Curves')

    # 下側子圖，顯示訓練和驗證準確度
    plt.subplot(2, 1, 2)
    plt.plot(train_accuracies, label='train_Accuracy')
    plt.plot(test_accuracies, label='validation_Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    # 調整子圖之間的間距，以確保它們不重疊
    plt.tight_layout()
    plt.savefig('./assign4_curve.png')
    #先用savefig是避免儲存到空白圖片 (坐標軸)
    plt.show()

start_training_and_save_model_and_save_picture()