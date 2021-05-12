from torch import nn
import torch.nn.functional as F


class ResNet(nn.Module):
    '''
    Residual Network architecture built for classification on CIFAR-100 dataset.
    See https://arxiv.org/pdf/1512.03385.pdf for overview of ResNets.

    Architecture: 
    (in filters, out filters) | (in spatial dim, out spatial dim)
    - Block 1:       (3, depth_1) | (32, 32)
    - Block 2: (depth_1, depth_1) | (32, 32)
    - Block 3: (depth_1, depth_2) | (32, 32)
    - Block 4: (depth_2, depth_2) | (32, 32)
    - Block 5: (depth_2, depth_3) | (32, 16)
    - Block 6: (depth_3, depth_3) |  (16, 16)
    - Global average pool + Linear
    '''
    
    def __init__(self, depth_1, depth_2, depth_3, dropout_p=0):
        super().__init__()
        self.dropout_p = dropout_p

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=depth_1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=depth_1)

        # Block 2
        self.conv2_1 = nn.Conv2d(in_channels=depth_1, out_channels=depth_1, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=depth_1)
        self.conv2_2 = nn.Conv2d(in_channels=depth_1, out_channels=depth_1, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=depth_1)

        # Block 3
        self.conv3_shortcut = nn.Conv2d(in_channels=depth_1, out_channels=depth_2, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=depth_1, out_channels=depth_2, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=depth_2)
        self.conv3_2 = nn.Conv2d(in_channels=depth_2, out_channels=depth_2, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(num_features=depth_2)

        # Block 4
        self.conv4_1 = nn.Conv2d(in_channels=depth_2, out_channels=depth_2, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(num_features=depth_2)
        self.conv4_2 = nn.Conv2d(in_channels=depth_2, out_channels=depth_2, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(num_features=depth_2)

        # Block 5
        self.conv5_shortcut = nn.Conv2d(in_channels=depth_2, out_channels=depth_3, kernel_size=1, stride=2, padding=0)
        self.conv5_1 = nn.Conv2d(in_channels=depth_2, out_channels=depth_3, kernel_size=3, stride=2, padding=1)
        self.bn5_1 = nn.BatchNorm2d(num_features=depth_3)
        self.conv5_2 = nn.Conv2d(in_channels=depth_3, out_channels=depth_3, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(num_features=depth_3)

        # Block 6
        self.conv6_1 = nn.Conv2d(in_channels=depth_3, out_channels=depth_3, kernel_size=3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(num_features=depth_3)
        self.conv6_2 = nn.Conv2d(in_channels=depth_3, out_channels=depth_3, kernel_size=3, stride=1, padding=1)
        self.bn6_2 = nn.BatchNorm2d(num_features=depth_3)

        self.linear = nn.Linear(in_features=depth_3, out_features=100)
    

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Block 2
        residual = F.relu(self.bn2_1(self.conv2_1(x)))
        residual = self.bn2_2(self.conv2_2(residual))
        x = x + residual
        x = F.relu(x)

        # Block 3
        residual = F.relu(self.bn3_1(self.conv3_1(x)))
        residual = self.bn3_2(self.conv3_2(residual))
        x = self.conv3_shortcut(x) + residual
        x = F.relu(x)

        # Block 4
        residual = F.relu(self.bn4_1(self.conv4_1(x)))
        residual = self.bn4_2(self.conv4_2(residual))
        x = x + residual
        x = F.relu(x)

        # Block 5
        residual = F.relu(self.bn5_1(self.conv5_1(x)))
        residual = self.bn5_2(self.conv5_2(residual))
        x = self.conv5_shortcut(x) + residual
        x = F.relu(x)

        # Block 6
        residual = F.relu(self.bn6_1(self.conv6_1(x)))
        residual = self.bn6_2(self.conv6_2(residual))
        x = x + residual
        x = F.relu(x)

        # Average pool + Linear
        x = F.avg_pool2d(x, kernel_size=16)
        x = x.view(x.size(0),-1)
        x = self.linear(F.dropout(x, self.dropout_p))
        return x
