import os
import numpy as np
import random

from torch.utils.data import Dataset
from scipy import ndimage

class KittiLoader(Dataset):
    def __init__(self, datapath, max_disparity, kernel, layers, batch_size):
        filelist = os.listdir(os.path.join(datapath, 'training', 'disp_noc_0'))
        self.idlist = [filename.split('_')[0] for filename in filelist]

        self.image_2_template = os.path.join(datapath, 'training', 'image_2', '{}_10.png')
        self.image_3_template = os.path.join(datapath, 'training', 'image_3', '{}_10.png')
        self.labels = os.path.join(datapath, 'training', 'disp_noc_0', '{}_10.png')

        self.max_disparity = max_disparity
        self.receptive_field_size = 37
        self.halfrecp = int(self.receptive_field_size/2)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, id):
        raw_disparity_image = np.array(ndimage.imread(self.labels.format(repr(id).zfill(6)), mode='I'), dtype=np.float64)
        disparity_image = raw_disparity_image/256

        valid_patches = 0
        while valid_patches < 1:
            x = random.randint(self.halfrecp+2+(self.max_disparity*2), disparity_image.shape[1]-3-self.halfrecp-self.max_disparity)
            y = random.randint(self.halfrecp+2, disparity_image.shape[0]-3-self.halfrecp)

            local_disparity = int(disparity_image[y,x])
            if local_disparity > 2 and local_disparity < self.max_disparity-2:
                valid_patches += 1

                raw_image_2 = np.array(ndimage.imread(self.image_2_template.format(repr(id).zfill(6)), mode='RGB'))
                raw_image_3 = np.array(ndimage.imread(self.image_3_template.format(repr(id).zfill(6)), mode='RGB'))

                image_2 = np.moveaxis((np.uint8(raw_image_2) - 128) / 256, -1, 0)
                image_3 = np.moveaxis((np.uint8(raw_image_3) - 128) / 256, -1, 0)

                patch_2 = image_2[:, y-self.halfrecp:y+self.halfrecp+1, x-self.halfrecp:x+self.halfrecp+1]
                patch_3 = image_3[:, y-self.halfrecp:y+self.halfrecp+1, x-local_disparity-118:x-local_disparity+119]

                if patch_2.shape != (3,37,37) or patch_3.shape != (3,37,237):
                    print('x:',x)
                    print('y:',y)
                    print('local disparity:', local_disparity)
                    print(patch_2.shape)
                    print(patch_3.shape)
                    patch_2 = np.ones((3,37,37))*0.001
                    patch_3 = np.ones((3,37,237))*0.001
                    local_disparity = 0          

        return (local_disparity, patch_2, patch_3)