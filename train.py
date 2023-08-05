import os
import torch

from model import Net
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np
import Handtrack_module as htm
import torch.nn as nn
import math
labels_name = os.listdir("dataset")
data = list()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = htm.handdetector(detectionCon=0.85)

class mydata(Dataset):
    def __init__(self,datas,labels):
        super(mydata,self).__init__()
        self.data = datas
        self.labels = labels
 
    def __getitem__(self,idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data,label
    def __len__(self):
        return len(self.data)
data = list()
label = list()
print(1)
c=1
tmp=0
def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)

for i in labels_name:

    tmp=(tmp+1)%101
    if(tmp==0):
        print(c)
        c+=1
    if c==141:
        break
    img = cv2.imread(os.path.join("dataset",i))
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray_img)
    gamma = math.log10(0.5)/math.log10(mean/255)
    img = gamma_trans(img,gamma)
    img = detector.findHands(img)
    lmlist = detector.findposition(img,draw =False)
    if(len(lmlist)==0):
        continue

    
    data.append(torch.tensor(np.array(lmlist).reshape(-1),dtype = torch.float32))
    if 'A'<=i[:1]<='Z':
        label.append(ord(i[:1])-65)
    elif i[:5]=="space":
        label.append(26)
    elif i[:3]=="del":
        label.append(27)

torch.save(data,'data.pt')
torch.save(label,'label.pt')
# data = torch.load('data.pt')
# label = torch.load('label.pt')

MyData = mydata(data,label)
myDataLoader = DataLoader(MyData,batch_size=20,shuffle=True,drop_last=True,num_workers=0)

net = Net()
net = net.to(device)
lossF=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = net.parameters(),lr = 1e-4)
totalLoss =0
for epoch in range(30):
    net.train(True)
    for id, (input,target) in enumerate(myDataLoader):

        input, target = input.to(device), target.to(device)

        net.zero_grad()
        outputs = net(input)
        loss = lossF(outputs,target)
        predictions = torch.argmax(outputs,dim=1)
        loss.backward()
        optimizer.step()
        if id %100 ==0:
            print(epoch,": ", id, "  ", loss.item())
        torch.cuda.empty_cache()
torch.save(net.state_dict(), "model_parameter.pkl")