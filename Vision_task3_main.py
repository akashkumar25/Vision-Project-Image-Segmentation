# Akash Kumar akku00001@teams.uni-saarland.de 7009735
# Harisree Kallakuri haka00001@teams.uni-saaland.de 7009317

#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from os.path import join as pjoin
import collections
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import imageio
import numpy as np
import pandas as pd
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import *

torch.cuda.set_device(4)
import torch
torch.cuda.empty_cache()

#Using Pytorch Cityscpaes Data Loader 
dst =torchvision.datasets.Cityscapes('./cityscapes',  split='train', mode='fine',target_type='semantic',
                                     transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((512,256))]),
                                     target_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((512,256))]))

data_loader = torch.utils.data.DataLoader(dst,batch_size=3,shuffle=True)

#Calling the model and loading the trained one
model= R2AttU_Net(img_ch=3,output_ch=34,t=2)
model.load_state_dict(torch.load('epoch-27.pt'))
model.eval()
model.cuda()



import torch.optim as optim
# loss function
loss_f = nn.CrossEntropyLoss()

# optimizer variable
opt = optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.00005)


#Training the model for 40 epochs and changing the mask type into long and reshaping as [batch_size, height, width]
epochs=40
for e in range(epochs):
  for i, d in enumerate(data_loader):
    img,lab= d
    img= (img*255).cuda()
    lab= (lab*255).long().reshape(lab.size()[0],512,256).cuda()
    pred_out= model.forward(img)
    loss= loss_f(pred_out,lab)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if i%10==0:
      print("epoch{}, iter{}, loss: {}".format(e, i, loss.data))
  torch.save(model.state_dict(), 'epochs-{}.pt'.format(e))






