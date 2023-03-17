import torch # Deep Learning env
import model # Neural network
import numpy as np # Counting
import cv2 # Data processing
from matplotlib import pyplot as plt # Plot



# To read file
f = open('./mnist/train_data.csv', 'r')

# 一次讀取每一行資料 
train_data = f.readlines()

# 讀完關閉
f.close()

# 訓練資料的長度，共1500筆
data_len = len(train_data)

# 訓練10次
epochs = 100

# 一個batch有100筆資料
batch = 100

# Learning rate
lr = 0.001

# 每一次要處理的資料量，以//令結果不會有float
batch_size = data_len//batch 

# LeNet model
net = model.Lenet()

# 使用Adam optimizer調整model參數讓Loss愈來愈小
opti = torch.optim.Adam(net.parameters(), lr=0.001)

# 每一個batch皆有9個0和1個1組成的array，以表示圖片的target
# torch.zeros：產生一組只有0和1矩陣
target = torch.zeros(batch, 10)

# (每一批輸入的batch數, 通道：1, 高：28 px, 寬：28 px）
model_input = torch.zeros(batch, 1, 28, 28)



 # 引用另一個.py需使用
if __name__ == '__main__': 

    train_loss = []

    # 訓練次數
    for i in range(epochs): 

        losses = []


        # 打散data_len的batch排列
        shuffled_batch = np.random.permutation(data_len) 

        

        # 要處理的資料量(15批)
        for j in range(batch_size):


            # 一批資料量的筆數(100筆)
            for k in range(batch):

                # j*batch：訓練幾次，k：100張之中的第幾張，二者相加即為當下總數
                a = j*batch+k 

                # 從資料夾讀圖片，要訓練的資料為打散過後的
                # 型態：「./Test/train_ 0.jpg 5」以.split('')區分空格左邊圖片路徑為[0]，右邊圖片編號為[1]
                r_img = cv2.imread('./mnist/' + train_data[shuffled_batch[a]].split(' ')[0], cv2.IMREAD_GRAYSCALE)

                # 在100張的第k張裡，輸入[0]，如此才是路徑對應的圖片。再用torch.from_numpy把讀到的圖片array轉成tensor
                model_input[k, 0] = torch.from_numpy(r_img)

                # 圖片的target則是輸入[1]，.split('')的圖片編號
                target[k, int(train_data[shuffled_batch[a]].split(' ')[1])] = 1


            # Loss function ((prediction_y - y)**2).mean()
            y = net(model_input)
            loss = ((y - target)**2).mean()



            
            # 歸零
            opti.zero_grad()


            # 反向傳播，計算當前loss的梯度
            loss.backward()

            
            # 根據loss梯度更新一個參數
            opti.step()
			
            # 取得每一次loss後的值
            losses.append(loss.item())
        
        # The average of loop i
        train_loss.append(np.mean(losses))
                              
   

    print(f' epoch: {epochs}\n batch size: {batch_size}\n loss: {loss}')

       
  
    plt.plot(list(range(epochs)), train_loss)
    plt.title('Loss function')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('./loss_values/loss_values_w.png', dpi=300)
    plt.show()