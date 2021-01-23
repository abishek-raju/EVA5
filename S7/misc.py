#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 19:24:57 2021

@author: rampfire
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def display_image(trainloader,classes):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    