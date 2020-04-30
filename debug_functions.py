#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 30 2020

@author: mcf
"""
import matplotlib.pyplot as plt
import numpy as np



def show_frame_image(frame, msg="Frame"):
    """
    For debuging, it is handy to view a frame as an image.
    :param frame: 2D numpy array to view as an image
    :param msg: Caption for the image
    :return: Nothing
    """
    a = frame.copy()
    a = np.expand_dims(a, axis=2)
    a = np.concatenate((a, a, a), axis=2)
    print(a.shape)
    plt.title(msg)
    plt.imshow(a)
    plt.show()
