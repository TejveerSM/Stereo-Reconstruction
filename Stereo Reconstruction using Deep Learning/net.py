import torch
import torch.nn as nn
import torch.nn.functional as F

# 4 layers
class SmallerNet(nn.Module):    
    def __init__(self, nChannel, max_disp):
        super(SmallerNet, self).__init__()                        
        self.l_max_disp = max_disp

        self.conv1 = nn.Conv2d(nChannel, 32, 5)    
        self.batchnorm1 = nn.BatchNorm2d(32, 1e-3)
        
        self.conv2 = nn.Conv2d(32, 32, 5) 
        self.batchnorm2 = nn.BatchNorm2d(32, 1e-3)
        
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.batchnorm3 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv4 = nn.Conv2d(64, 64, 5) 
        self.batchnorm4 = nn.BatchNorm2d(64, 1e-3)

        self.logsoftmax = nn.LogSoftmax(dim=0) 
            
    def forward_pass(self, x):
        x = self.conv1(x)                
        x = F.relu(self.batchnorm1(x))
        
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        
        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        
        x = self.conv4(x)
        x = self.batchnorm4(x)

        return x
             
    def forward(self, left_patch, right_patch):
        left_patch = self.forward_pass(left_patch)
        right_patch = self.forward_pass(right_patch)

        left_patch = left_patch.view(left_patch.size(0),1,64)        
        right_patch = right_patch.squeeze().view(right_patch.size(0),64,self.l_max_disp)

        inner_product = left_patch.bmm(right_patch).view(right_patch.size(0),self.l_max_disp)
        inner_product = self.logsoftmax(inner_product)

        return left_patch, right_patch, inner_product

# 9 layers
class LargerNet(nn.Module):    
    def __init__(self, nChannel, max_disp):
        super(LargerNet, self).__init__()                        
        self.l_max_disp = max_disp

        self.conv1 = nn.Conv2d(nChannel, 32, 5)    
        self.batchnorm1 = nn.BatchNorm2d(32, 1e-3)
        
        self.conv2 = nn.Conv2d(32, 32, 5) 
        self.batchnorm2 = nn.BatchNorm2d(32, 1e-3)
        
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.batchnorm3 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv4 = nn.Conv2d(64, 64, 5) 
        self.batchnorm4 = nn.BatchNorm2d(64, 1e-3) 
        
        self.conv5 = nn.Conv2d(64, 64, 5)
        self.batchnorm5 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv6 = nn.Conv2d(64, 64, 5)
        self.batchnorm6 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv7 = nn.Conv2d(64, 64, 5)
        self.batchnorm7 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv8 = nn.Conv2d(64, 64, 5)
        self.batchnorm8 = nn.BatchNorm2d(64, 1e-3)       
            
        self.conv9 = nn.Conv2d(64, 64, 5)
        self.batchnorm9 = nn.BatchNorm2d(64, 1e-3)  

        self.logsoftmax = nn.LogSoftmax(dim=0)
            
    def forward_pass(self, x):
        x = self.conv1(x)                
        x = F.relu(self.batchnorm1(x))
        
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        
        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        
        x = self.conv4(x)
        x = F.relu(self.batchnorm4(x))
        
        x = self.conv5(x)
        x = F.relu(self.batchnorm5(x))
        
        x = self.conv6(x)
        x = F.relu(self.batchnorm6(x))
        
        x = self.conv7(x)
        x = F.relu(self.batchnorm7(x))
        
        x = self.conv8(x)
        x = F.relu(self.batchnorm8(x))
        
        x = self.conv9(x)
        x = self.batchnorm9(x)

        return x
             
    def forward(self, left_patch, right_patch):
        left_patch = self.forward_pass(left_patch)
        right_patch = self.forward_pass(right_patch)

        left_patch = left_patch.view(left_patch.size(0),1,64)        
        right_patch = right_patch.squeeze().view(right_patch.size(0),64,self.l_max_disp)

        inner_product = left_patch.bmm(right_patch).view(right_patch.size(0),self.l_max_disp)
        inner_product = self.logsoftmax(inner_product)

        return left_patch, right_patch, inner_product


class TestNet(nn.Module):    
    def __init__(self, nChannel):
        super(TestNet, self).__init__()                
        self.pad = nn.ReflectionPad2d(18)
        
        self.conv1 = nn.Conv2d(nChannel, 32, 5)
        self.batchnorm1 = nn.BatchNorm2d(32, 1e-3)
        
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.batchnorm2 = nn.BatchNorm2d(32, 1e-3)
        
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.batchnorm3 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.batchnorm4 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv5 = nn.Conv2d(64, 64, 5)
        self.batchnorm5 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv6 = nn.Conv2d(64, 64, 5)
        self.batchnorm6 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv7 = nn.Conv2d(64, 64, 5)
        self.batchnorm7 = nn.BatchNorm2d(64, 1e-3)
        
        self.conv8 = nn.Conv2d(64, 64, 5)
        self.batchnorm8 = nn.BatchNorm2d(64, 1e-3)     
            
        self.conv9 = nn.Conv2d(64, 64, 5)
        self.batchnorm9 = nn.BatchNorm2d(64, 1e-3)
        
    def forward(self, x):
        
        x = self.pad(x)                
        x = self.conv1(x)                
        x = F.relu(self.batchnorm1(x))
        
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        
        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        
        x = self.conv4(x)
        x = F.relu(self.batchnorm4(x))
        
        x = self.conv5(x)
        x = F.relu(self.batchnorm5(x))
        
        x = self.conv6(x)
        x = F.relu(self.batchnorm6(x))
        
        x = self.conv7(x)
        x = F.relu(self.batchnorm7(x))
        
        x = self.conv8(x)
        x = F.relu(self.batchnorm8(x))
        
        x = self.conv9(x)
        x = self.batchnorm9(x)
                
        return x