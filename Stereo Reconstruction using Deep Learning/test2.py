import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import IPython.display
import config as cfg

disp_range = 128

class Net(nn.Module):    
    def __init__(self, nChannel):
        super(Net, self).__init__()                
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

def disparity_to_color(I):
    
    _map = np.array([[0,0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174], 
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]]
                   )      
    max_disp = 1.0*I.max()
    I = np.minimum(I/max_disp, np.ones_like(I))
    
    A = I.transpose()
    num_A = A.shape[0]*A.shape[1]
    
    bins = _map[0:_map.shape[0]-1,3]    
    cbins = np.cumsum(bins)    
    cbins_end = cbins[-1]
    bins = bins/(1.0*cbins_end)
    cbins = cbins[0:len(cbins)-1]/(1.0*cbins_end)
    
    A = A.reshape(1,num_A)            
    B = np.tile(A,(6,1))        
    C = np.tile(np.array(cbins).reshape(-1,1),(1,num_A))
       
    ind = np.minimum(sum(B > C),6)
    bins = 1/bins
    cbins = np.insert(cbins, 0,0)
    
    A = np.multiply(A-cbins[ind], bins[ind])   
    K1 = np.multiply(_map[ind,0:3], np.tile(1-A, (3,1)).T)
    K2 = np.multiply(_map[ind+1,0:3], np.tile(A, (3,1)).T)
    K3 = np.minimum(np.maximum(K1+K2,0),1)
    
    return np.reshape(K3, (I.shape[1],I.shape[0],3)).T

if __name__ == '__main__':
    model = Net(3)
    model.load_state_dict(torch.load(cfg.SAVE_PATH.format(cfg.TEST_EPOCH)))
    model.eval()

    left_image_fn = 'KITTI/testing/image_2/000022_10.png'
    right_image_fn = 'KITTI/testing/image_3/000022_10.png'   

    l_img = 255*transforms.ToTensor()(Image.open(left_image_fn))
    r_img = 255*transforms.ToTensor()(Image.open(right_image_fn))

    l_img = (l_img - l_img.mean())/(l_img.std())
    r_img = (r_img - r_img.mean())/(r_img.std())

    img_h = l_img.size(1)
    img_w = l_img.size(2)

    l_img = l_img.view(1, l_img.size(0), l_img.size(1), l_img.size(2))
    r_img = r_img.view(1, r_img.size(0), r_img.size(1), r_img.size(2))

    with torch.no_grad():
        left_feat = model(Variable(l_img))
        right_feat = model(Variable(r_img))

    output = torch.Tensor(img_h, img_w, disp_range).zero_()

    end_id = img_w-1
    for loc_idx in range(disp_range):
        l = left_feat[:,:,:,loc_idx:end_id]
        r = right_feat[:,:,:,0:end_id-loc_idx]         
        p = torch.mul(l,r)  
        q = torch.sum(p,1)              
        output[:,loc_idx:end_id,loc_idx] = q.data.view(q.data.size(1), q.data.size(2))                                    

    max_disp, pred = torch.max(output,2)

    pred = pred.view(output.size(0),output.size(1))

    color_map = disparity_to_color(pred.float().numpy())
    final_image = Image.fromarray(np.uint8(255*np.transpose(color_map, axes=[1,2,0])))
    final_image.save('demo_img.png')
    print('done')