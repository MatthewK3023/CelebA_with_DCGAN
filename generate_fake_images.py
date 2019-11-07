#!/usr/bin/env python
# coding: utf-8

# In[20]:


import warnings
warnings.filterwarnings('ignore')
import sys
import os
import argparse
import os
import numpy as np
import math
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pickle as pkl
import time
import itertools
import cv2
import PIL.Image as Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
# config.gpu_options.visible_device_list = '2'
# session = tf.compat.v1.Session(config=config)#tf.compat.v1.keras.backend.set_session(session)
sys.argv

import helper
data_dir = './data'
helper.download_extract(data_dir)


# In[21]:


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)


# In[22]:


z_dim= 100
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # state size = 512*4*4
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # state size = 256*8*8
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # state size = 128*16*16
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # state size = 64*32*32
            
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size = 3*64*64
        )
    
    def forward(self, x):
        return self.network(x)


# In[23]:


model = torch.load('GAN_model/GANnet_epoch50.pkl')


# In[38]:


# Generate 4500 images
n = 900
steps = 5
for step in range(steps):
    fixed_noise = torch.randn(n, 100, 1, 1, device=device)
    fake_p = model(fixed_noise).detach().cpu()
    for i in range(n):
        save_image(fake_p[i], './fake_images/img%06d.png'%(i+(900*step)))
    del fake_p


# In[39]:


len(os.listdir('./fake_images'))


# In[47]:


def output_fig(images_array, file_name):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)


# In[50]:


for i in range(500):
    width = height = 64
    generated_images = helper.get_batch(glob('fake_images/*.png')[0+(i*9):9+(i*9)], 64, 64, 'RGB')
    
    output_fig(generated_images, file_name="./fake_images_9x9/img%03d.png"%i)
    del generated_images


# In[48]:


output_fig(generated_images)


# In[ ]:


"./fake_images_9x9/img%03d.png"%

