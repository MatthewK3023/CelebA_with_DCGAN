#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import PIL.Image as Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.utils as vutils

import torch.nn as nn
import torch.nn.functional as F
import torch
sys.argv

import helper
data_dir = './data'
helper.download_extract(data_dir)


# In[2]:


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


cuda = True if torch.cuda.is_available() else False
cuda


# In[4]:


img_size = 64
isCrop = False
z_dim = 100

batch_size = 128
lr = 0.0002
train_epoch = 20

data_dir = './cropped_images64'
transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_set = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

batch = next(iter(train_loader))[0]

nrows=4
ncols=4
fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=True)
for i in range(nrows):
    for j in range(ncols):
        img = batch[nrows * i + j].numpy()
        img = img.transpose((1, 2, 0))
        axes[i, j].imshow(img)

plt.show()


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.network(x)



# create the generator
netG = Generator().to(device)

# apply the weight_init function to randomly initialize all the weights
netG.apply(weight_init)

# print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 128*32*32
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 128*16*16
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 256*8*8
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 512*4*4
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # state size 1*1*1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


netD = Discriminator().to(device)

# initializing the weights
netD.apply(weight_init)

print(netD)

# inititalize the BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_labels = 1
fake_labels = 0

# Setup Adam optimizers for both G and D
lr = 0.0002

g_optim = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
d_optim = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))


img_list = []
g_losses = []
d_losses = []
iters = 0
epochs = 50
print_every = 50
save_img_every = 500


print('starting training...')

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        
        # train with all-real batch
        netD.zero_grad()
        # Format batch
        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_labels, device=device)
        
        # forward pass real batch through D
        output = netD(real).view(-1)
        
        # calculate loss on all-real batch
        d_loss_real = criterion(output, label)
        
        # calculate gradients for D in backward pass
        d_loss_real.backward()
        d_x = output.mean().item()
        
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        # generate fake images with G
        fake = netG(noise)
        label.fill_(fake_labels)
        # classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # calculate D's loss on the all-fake batch
        d_loss_fake = criterion(output, label)
        # Calculate the gradients for this batch
        d_loss_fake.backward()
        d_g_z1 = output.mean().item()
        # add the gradients from the all-real and all-fake batches
        d_loss = d_loss_fake + d_loss_real
        # update D
        d_optim.step()
        
        netG.zero_grad()
        label.fill_(real_labels)  # fake labels are real for generator cost
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        
        # Calculate G's loss based on this output
        g_loss = criterion(output, label)
        # Calculate gradients for G
        g_loss.backward()
        d_g_z2 = g_loss.mean().item()
        # Update G
        g_optim.step()
        
        if i % print_every == 0:
            print('[{}/{}][{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f}'.format(
                epoch, epochs, i, len(train_loader), d_loss.item(), g_loss.item(), d_x, d_g_z1, d_g_z2))
        
        # save losses for plotting
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        # Output training stats
        if i % save_img_every == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1
        
        if (epoch+1) % 5 == 0:
            torch.save(netG.state_dict(), './GAN_model/GAN_epoch%d'%(epoch+1))
            torch.save(netG, './GAN_model/GANnet_epoch%d.pkl'%(epoch+1))

print('end of training...')



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses,label="G")
plt.plot(d_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()



fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


real_batch = next(iter(train_loader))


