import torch.nn as nn
import torch
from torchsummary import summary
class Net(nn.Module):
    def __init__(self) :
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=63,out_features=512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128,out_features=28),
            nn.Softmax(dim=1)
        )
    def forward(self,input):
        output = self.model(input)
        return output
net = Net()
if torch.cuda.is_available():
    net.cuda()
