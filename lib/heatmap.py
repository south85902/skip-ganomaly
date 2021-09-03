"""
Losses
"""
# pylint: disable=C0301,C0103,R0902,R0915,W0221,W0622


##
# LIBRARIES
import torch
import cv2
import numpy as np

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def save_heatmap(des, heatmap):
    cv2.imwrite(des, heatmap)


##

