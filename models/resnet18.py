import torch.nn as nn
import torch.nn.functional as F


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.base_channels = 64
        
        self.prepblock1 = nn.Sequential(
            nn.Conv2d(3, self.base_channels, 3, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.Dropout(0.2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.Dropout(0.2),
        )

        self.mixer2 = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels*2, 1, stride=2),
            nn.BatchNorm2d(self.base_channels*2)
        )
        self.prepblock2 = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels*2),
            nn.Dropout(0.2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.base_channels*2, self.base_channels*2, 3, padding=1),
            nn.BatchNorm2d(self.base_channels*2),
            nn.Dropout(0.2),
        )

        self.mixer3 = nn.Sequential(
            nn.Conv2d(self.base_channels*2, self.base_channels*4, 1, stride=2),
            nn.BatchNorm2d(self.base_channels*4)
        )
        self.prepblock3 = nn.Sequential(
            nn.Conv2d(self.base_channels*2, self.base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels*4),
            nn.Dropout(0.2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(self.base_channels*4, self.base_channels*4, 3, padding=1),
            nn.BatchNorm2d(self.base_channels*4),
            nn.Dropout(0.2),
        )

        self.mixer4 = nn.Sequential(
            nn.Conv2d(self.base_channels*4, self.base_channels*8, 1, stride=2),
            nn.BatchNorm2d(self.base_channels*8)
        )
        self.prepblock4 = nn.Sequential(
            nn.Conv2d(self.base_channels*4, self.base_channels*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels*8),
            nn.Dropout(0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(self.base_channels*8, self.base_channels*8, 3, padding=1),
            nn.BatchNorm2d(self.base_channels*8),
            nn.Dropout(0.2),
        )
        self.Relu = nn.ReLU()

        self.gap = nn.AvgPool2d(kernel_size=3)
        self.fc = nn.Linear(self.base_channels*8, 10)

    def forward(self, x):
        x = self.prepblock1(x)

        res1 = self.block1(self.Relu(self.block1(x)))
        x = self.Relu(x + res1)
        res2 = self.block1(self.Relu(self.block1(x)))
        x = self.Relu(x + res2)

        res3 = self.block2(self.Relu(self.prepblock2(x)))
        x = self.mixer2(x)
        x = self.Relu(x + res3)
        res4 = self.block2(self.Relu(self.block2(x)))
        x = self.Relu(x + res4)

        res5 = self.block3(self.Relu(self.prepblock3(x)))
        x = self.mixer3(x)
        x = self.Relu(x + res5)
        res6 = self.block3(self.Relu(self.block3(x)))
        x = self.Relu(x + res6)

        res3 = self.block4(self.Relu(self.prepblock4(x)))
        x = self.mixer4(x)
        x = self.Relu(x + res3)
        res4 = self.block4(self.Relu(self.block4(x)))
        x = x + res4
    
        x = self.gap(x)

        x = x.view(-1, self.base_channels*8)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)



