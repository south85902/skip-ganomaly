"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from lib.data.datasets import get_cifar_anomaly_dataset
from lib.data.datasets import get_mnist_anomaly_dataset

# add padding ===================
# 參考 https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
import numpy as np
from torchvision.transforms.functional import pad

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return pad(image, padding, (0, 0, 0), 'constant')
# add padding ===================

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = '../dataSet/{}'.format(opt.dataset)

    ## CIFAR
    if opt.dataset in ['cifar10']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_ds = CIFAR10(root='../dataSet', train=True, download=True, transform=transform)
        valid_ds = CIFAR10(root='../dataSet', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_cifar_anomaly_dataset(train_ds, valid_ds, train_ds.class_to_idx[opt.abnormal_class])

    ## MNIST
    elif opt.dataset in ['mnist']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])


        train_ds = MNIST(root='../dataSet', train=True, download=True, transform=transform)
        valid_ds = MNIST(root='../dataSet', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class))

    # FOLDER
    else:
        # transform = transforms.Compose([transforms.Resize(opt.isize),
        #                                 transforms.CenterCrop(opt.isize),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        if opt.no_padding:
            # transform = transforms.Compose([transforms.Resize([opt.isize, opt.isize]),
            #                                 transforms.ToTensor(),
            #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
            transform = transforms.Compose([transforms.Resize([opt.isize, 32]),
                                            SquarePad(),
                                            #transforms.Resize([opt.isize, opt.isize]),
                                            transforms.CenterCrop(opt.isize),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        else:
            # add padding
            transform = transforms.Compose([SquarePad(),
                                            transforms.Resize(opt.isize),
                                            transforms.CenterCrop(opt.isize),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
        valid_ds = ImageFolder(os.path.join(opt.dataroot, 'val'), transform)
        test_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl, test_dl)
