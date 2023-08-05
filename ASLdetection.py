import cv2
import Handtrack_module as htm
import numpy as np
import torch
from model import Net
net = Net()
net.load_state_dict(torch.load("model_parameters/model_parameter.pkl",map_location=torch.device('cpu')))

video = cv2.VideoCapture(0)
detector = htm.handdetector(detectionCon=0.85)
# img = detector.findHands(img)
# lmlist = detector.findposition(img,draw =False)


video.set(3,800)
video.set(4,800)
t =0 
fingers = []
net.eval()
c=0
pastchar = None
curchar:chr
while True:
    
    success, img = video.read()
    
    
    if c<15:
        c+=1
    else:
        img = detector.findHands(img)
        lmlist = detector.findposition(img,draw =False)
        if(len(lmlist)!=0):
            lmlist = np.array(lmlist).reshape(-1,63)
            outputs = net(torch.tensor(lmlist,dtype = torch.float32))
            if(0<=torch.argmax(outputs,dim=1)<=25):
                curchar = chr(torch.argmax(outputs,dim=1)+65)
            elif torch.argmax(outputs,dim=1)==26:
                curchar = "space"
            elif torch.argmax(outputs,dim=1)==27:
                curchar = "del"

        else:
            curchar  = "nothing"
        if curchar!=pastchar:
            print(curchar)
            pastchar = curchar
        c=0
    cv2.imshow("Img",img)
    cv2.waitKey(1)

'''
I can set a threshold. When neural network determines which hand should it be, it cools down for 1 second. 





# '''