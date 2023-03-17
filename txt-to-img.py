import numpy as np
import cv2

# 打開txt檔
f = open('D:/python/mnist/Train.txt', 'r')
line = f.readlines()
f.close()

# 用for 寫要轉換的張數(500)
# Labels
pic = []
for i in range(1500):
    a = line[i].split(' ')
    b = a[1:11] # 從第1張到第10張(第0張是|Labels、第11張是|features故不列入)
    max_index = 0
    for j in range(len(b)):
        if b[j] > b[max_index]: # 找到>0的順序是第幾個
            max_index = j

# Features用np.uint8 
    c = np.uint8(a[12:])
    
# cv2的channel在第三個，28*28 pixel
    c = c.reshape((28, 28, 1))
    cv2.imwrite('./train/train_' + str(i) + '.jpg', c) # +str(i)+ 幫檔名加上流水編號

    pic.append('./train/train_' + str(i) + '.jpg ' + str(max_index))
    
with open('./train_data.csv', 'w') as tmp_file: # output一個.csv 以利Neural Network處理
    for i in pic:
        tmp_file.write(i + '\n')

