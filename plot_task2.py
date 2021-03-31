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

torch.cuda.set_device(2)
import torch
torch.cuda.empty_cache()

dst =torchvision.datasets.Cityscapes('./cityscapes',  split='train', mode='fine',target_type='semantic',
                                     transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((512,256))]),
                                     target_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((512,256))]))

data_loader = torch.utils.data.DataLoader(dst,batch_size=2,shuffle=True)


model= R2U_Net(img_ch=3,output_ch=34,t=2)
model.load_state_dict(torch.load('epochs-4.pt'))
model.eval()
model.cuda()


torch.manual_seed(10)
val= iter(data_loader)
for e in range(10):
  fig= plt.figure(figsize=(10,10))
  image,mask= next(val)
  image= image.cuda()
  mask= mask.cuda().detach().cpu()
  preds= model(image)
  preds= preds.detach().cpu()
  image= image.cpu()
  fig1= fig.add_subplot(131)
  plt.imshow(image[0].transpose(0,2).transpose(0,1).numpy())
  fig1.title.set_text("Image")
  fig1.axis("off")
  fig2= fig.add_subplot(132)
  plt.imshow(mask[0].transpose(0,2).transpose(0,1).numpy())
  fig2.title.set_text("Ground_Truth")
  fig2.axis("off")
  fig3= fig.add_subplot(133)
  plt.imshow(preds.argmax(1)[0].numpy())
  fig3.title.set_text("Prediction")
  fig3.axis("off")
  plt.show()
  plt.savefig('epoch{}'.format(e))

  '''
  print(image.transpose(0,2).transpose(0,1).numpy())
  print(mask.shape)
  print(preds.shape)'''