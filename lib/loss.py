"""
Losses
"""
# pylint: disable=C0301,C0103,R0902,R0915,W0221,W0622


##
# LIBRARIES
import torch
from lib import pytorch_ssim
##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)

def ssim_loss(input, target):
    ssim_loss = pytorch_ssim.SSIM(window_size=17)
    return -ssim_loss(input, target)

def ssiml1_loss(input, target):
    s_loss = pytorch_ssim.SSIM(window_size=11)
    s = -s_loss(input, target)
    l = torch.mean(torch.abs(input - target))
    t = s+l
    return t
