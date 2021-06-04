import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)# 4*84*84-->32*20*20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)# 32*20*20-->64*9*9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)# 64*9*9-->64*7*7

        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() #/ 255
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class DuelingDQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(7 * 7 * 64, 256) # 256 or 512
        self.fc2_adv = nn.Linear(256, n_actions) # 256 or 512

        self.fc1_val = nn.Linear(7 * 7 * 64, 256) # 256 or 512
        self.fc2_val = nn.Linear(256, 1) # 256 or 512

        self.n_actions = n_actions

    def forward(self, x):
        x = x.float() # /1.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        adv = F.relu(self.fc1_adv(x.view(x.size(0), -1)))
        adv = self.fc2_adv(adv)  #  shape=[1,4]

        val = F.relu(self.fc1_val(x.view(x.size(0), -1))) # shape=[1,512]
        val = self.fc2_val(val).expand(x.size(0), self.n_actions) # shape=[1,4]
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        return x