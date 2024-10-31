import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio
import cv2
from match_images import matching


#load image
class IRDST_TrainSetLoader(Dataset):
    def __init__(self, root, fullSupervision=True, align=True):

        txtpath = root + 'img_idx/train_IRDST-simulation.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.root = root
        self.fullSupervision = fullSupervision
        self.align = align
        self.train_mean = 106.8523
        self.train_std = 56.9243

    def __getitem__(self, index):
        imgfolder = 'images'
        img_path = os.path.join(self.root, imgfolder, self.imgs_arr[index] + 'bmp')
        seq = self.imgs_arr[index].split('/')[0]
        frame = int(self.imgs_arr[index].split('/')[1])
        img_ori = cv2.imread(img_path)
        if np.dim(img_ori) == 3:
            img_ori = img_ori[:,:,0]
        img = np.expand_dims(img_ori.astype(np.float32), axis=0)

        for i in range(1,5):
            img_hispath = os.path.join(self.root, imgfolder, seq, str(max(frame-i, 1)) + '.bmp')
            img_his = cv2.imread(img_hispath)
            if np.ndim(img_his) == 3:
                img_his= img_his[:,:,0]
            if self.align:
                img_his = matching(img_his, img_ori)
            img_his = np.expand_dims(img_his.astype(np.float32), axis=0)
            img= np.concatenate((img_his, img), axis=0)


        # Read Label/Tgt
        label_path = img_path.replace(imgfolder, 'masks')
        label = cv2.imread(label_path).astype(np.float32) / 255.0

        # Mix preprocess
        img = (img - self.train_mean)/self.train_std
        img = torch.unsqueeze(torch.from_numpy(img), 0)
        label = torch.unsqueeze(torch.from_numpy(label), 0)

        [_, m, n] = np.shape(label)
        return img, label, m, n


    def __len__(self):
        return len(self.imgs_arr)



class IRDST_TestSetLoader(Dataset):
    def __init__(self, root, fullSupervision=True, align=True):

        txtpath = root + 'img_idx/test_IRDST-simulation.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.root = root
        self.fullSupervision = fullSupervision
        self.align = align
        self.train_mean = 106.8523
        self.train_std = 56.9243

    def __getitem__(self, index):
        imgfolder = 'images'
        img_path = os.path.join(self.root, imgfolder, self.imgs_arr[index] + 'bmp')
        seq = self.imgs_arr[index].split('/')[0]
        frame = int(self.imgs_arr[index].split('/')[1])
        img_ori = cv2.imread(img_path)
        if np.dim(img_ori) == 3:
            img_ori = img_ori[:,:,0]
        img = np.expand_dims(img_ori.astype(np.float32), axis=0)

        for i in range(1,5):
            img_hispath = os.path.join(self.root, imgfolder, seq, str(max(frame-i, 1)) + '.bmp')
            img_his = cv2.imread(img_hispath)
            if np.ndim(img_his) == 3:
                img_his = img_his[:,:,0]
            if self.align:
                img_his = matching(img_his, img_ori)
            img_his = np.expand_dims(img_his.astype(np.float32), axis=0)
            img= np.concatenate((img_his, img), axis=0)


        # Read Label/Tgt
        label_path = img_path.replace(imgfolder, 'masks')
        label = cv2.imread(label_path).astype(np.float32) / 255.0

        # Mix preprocess
        img = (img - self.train_mean)/self.train_std
        img = torch.unsqueeze(torch.from_numpy(img), 0)
        label = torch.unsqueeze(torch.from_numpy(label), 0)

        [_, m, n] = np.shape(label)
        return img, label, m, n


    def __len__(self):
        return len(self.imgs_arr)
