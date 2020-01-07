import os
import random
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from PIL import Image

import torch
from torch.optim import Adagrad
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import config as cfg
from net import LargerNet
from dataloader import KittiLoader
from loss import px3_loss

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    kitti_dataset = dataset=KittiLoader(cfg.KITTIPATH, cfg.MAX_DISPARITY, cfg.KERNEL, cfg.CONV_LAYERS, cfg.BATCH_SIZE)
    dataloader = DataLoader(dataset=kitti_dataset,
                            batch_size=cfg.BATCH_SIZE,
                            shuffle=False)
    model = LargerNet(3,201)

    if torch.cuda.device_count():
        model = model.cuda()
        loss_weights = torch.tensor(cfg.LOSS_WEIGHTS, dtype=torch.float64).cuda()
    else:
        loss_weights = torch.tensor(cfg.LOSS_WEIGHTS, dtype=torch.float64)
    
    model.train()

    optimizer = Adagrad(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    for epoch in range(cfg.MAX_EPOCHS):
        train_loss = 0
        
        for _iter in tqdm(range(cfg.EPOCH_ITERS)):
            batch_iterator = iter(dataloader)  
            optimizer.zero_grad()
            
            (gt, patch_2, patch_3) = next(batch_iterator)

            if torch.cuda.device_count():
                gt = gt.cuda()
                patch_2 = patch_2.cuda()
                patch_3 = patch_3.cuda()

            if patch_2.size() == (16,3,37,37) and patch_3.size() == (16,3,37,237):
                _,_,softmax_scores = model(Variable(patch_2), Variable(patch_3))
                loss = px3_loss(softmax_scores, gt, loss_weights)
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item()
            
        print('Epoch: {} Loss: {}'.format(epoch, train_loss/cfg.EPOCH_ITERS/cfg.BATCH_SIZE))
        torch.save(model.state_dict(), cfg.SAVE_PATH.format(epoch))

    print('Done.')