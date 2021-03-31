# Akash Kumar akku00001@teams.uni-saarland.de 7009735
# Harisree Kallakuri haka00001@teams.uni-saaland.de 7009317

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
from model_task2 import *
from metric import *

torch.cuda.set_device(6)
import torch
torch.cuda.empty_cache()

dst =torchvision.datasets.Cityscapes('./cityscapes',  split='train', mode='fine',target_type='semantic',
                                     transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((512,256))]),
                                     target_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((512,256))]))

data_loader = torch.utils.data.DataLoader(dst,batch_size=2,shuffle=True)

# loss function
loss_f = nn.CrossEntropyLoss()

model= R2AttU_Net(img_ch=3,output_ch=34,t=2)
model.load_state_dict(torch.load('epochs-4.pt'))
model.eval()
model.cuda()

epochs =1
for e in range(epochs):
  for i, d in enumerate(data_loader):
    # your code goes here
    img,lab= d
    img= (img*255).cuda()
    lab= (lab*255).long().reshape(lab.size()[0],512,256).cuda()
    pred_out= model(img)
    loss= loss_f(pred_out,lab)
    if i%10==0:
      print("epoch{}, iter{}, loss: {}".format(e, i, loss.data))



TP,TN,FP,FN= confusion_matrix(lab[0],pred_out.argmax(1)[0])

print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP,TN,FP,FN))

accuracy= (TP+TN)/(TP+TN+FP+FN)
sensitivity= TP/(TP+FN)
specificity= TN/(TN+FP)

f1,jac= evaluation_metrics(lab[0],pred_out.argmax(1)[0])

print("accuracy: {}, sensitivity: {}, specificity: {}, f1-score: {}, jaccard-score: {}".format(accuracy,sensitivity,specificity,f1,jac))
