import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio


#load image
class CSIG_TrainSetLoader(Dataset):
    def __init__(self, root, fullSupervision=False):

        txtpath = root + 'train_list.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.root = root        # dataset/CSIG/
        self.fullSupervision = fullSupervision
        self.train_mean = 111.47
        self.train_std = 22.43

    def extract_number(self, filename):
        return int(os.path.splitext(filename)[0])

    def __getitem__(self, index):
        img_path_mix = self.imgs_arr[index]     # 'dataset/CSIG/train\000001\img\000001.png'

        img_path_mix = img_path_mix.replace('\\', '/')
        dir_path, filename = os.path.split(img_path_mix)  # 'dataset/CSIG/train/000001/img', '000007.png'
        folder_name = os.path.basename(os.path.dirname(dir_path))  # '000001'
        img_dir_root = os.path.dirname(dir_path)  # 'dataset/CSIG/train/000001'

        file_basename = os.path.splitext(filename)[0]
        ext = '.png'
        img_path_tgt = os.path.join(img_dir_root, 'mask', file_basename + ext)
        
        LabelData_Img = Image.open(img_path_tgt)
        LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
        LabelData_Img[LabelData_Img > 0] = 1.0

        img_dir = os.path.join(img_dir_root, 'img')
        img_list = sorted(os.listdir(img_dir), key=self.extract_number)

        try:
            idx = img_list.index(filename)
        except ValueError:
            raise FileNotFoundError(f"{filename} not found in {img_dir}")

        if idx >= 4:
            indices = range(idx - 4, idx + 1)
        else:
            indices = range(idx, idx + 5)
            indices = indices[::-1]

        MixData_Img = []
        for i in indices:
            if i >= len(img_list):
                break
            img_path = os.path.join(img_dir, img_list[i])
            # print(img_path)
            img = Image.open(img_path)
            img_np = np.array(img, dtype=np.float32)
            MixData_Img.append(img_np)

        MixData_Img = np.stack(MixData_Img, axis=0).astype(np.float32)
        MixData_Img = (MixData_Img - self.train_mean) / self.train_std
        MixData = torch.from_numpy(MixData_Img)
        MixData_out = torch.unsqueeze(MixData[-5:,:,:], 0)

        [m_L, n_L] = np.shape(LabelData_Img)

        if m_L == 256 and n_L == 256:
            # Tgt preprocess
            LabelData = torch.from_numpy(LabelData_Img)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            return MixData_out, TgtData_out, m_L, n_L,

        # elif m_L < 256 and n_L < 256:
        #     # Tgt preprocess
        #     [n, t, m_M, n_M] = shape(MixData_out)
        #     LabelData_Img_1 = np.zeros([1024,1024])
        #     LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img
        #     LabelData = torch.from_numpy(LabelData_Img_1)
        #     TgtData_out = torch.unsqueeze(LabelData, 0)
        #     MixData_out_1 = torch.zeros([n,t,1024,1024])
        #     MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out
        #     return MixData_out_1, TgtData_out, m_L, n_L

        elif m_L == 1024 and n_L == 1024:
            ys, xs = np.where(LabelData_Img > 0)
            if len(ys) == 0:
                top = random.randint(0, m_L - 256)
                left = random.randint(0, n_L - 256)
            elif len(ys) == 1:
                idx = 0
                y, x = ys[idx], xs[idx]
                top = y - random.randint(0, 128)
                left = x - random.randint(0, 128)
                top = max(0, top)
                left = max(0, left)
                if top + 256 > 1024:
                    top = 1024 - 256
                if left + 256 > 1024:
                    left = 1024 - 256
            else:
                idx = random.randint(0, len(ys) - 1)
                y, x = ys[idx], xs[idx]
                top = y - random.randint(0, 128)
                left = x - random.randint(0, 128)
                top = max(0, top)
                left = max(0, left)
                if top + 256 > 1024:
                    top = 1024 - 256
                if left + 256 > 1024:
                    left = 1024 - 256
            [n, t, m_M, n_M] = shape(MixData_out)

            LabelData_Img_1 = LabelData_Img[top:top + 256, left:left + 256]
            LabelData = torch.from_numpy(LabelData_Img_1)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            MixData_out_1 = MixData_out[:, :, top:top + 256, left:left + 256]

            return MixData_out_1, TgtData_out, m_L, n_L

        elif m_L == 512 and n_L == 640:
            ys, xs = np.where(LabelData_Img > 0)
            if len(ys) == 0:
                top = random.randint(0, m_L - 256)
                left = random.randint(0, n_L - 256)
            elif len(ys) == 1:
                idx = 0
                y, x = ys[idx], xs[idx]
                top = y - random.randint(0, 128)
                left = x - random.randint(0, 128)
                top = max(0, top)
                left = max(0, left)
                if top + 256 > 512:
                    top = 512 - 256
                if left + 256 > 640:
                    left = 640 - 256
            else:
                idx = random.randint(0, len(ys) - 1)
                y, x = ys[idx], xs[idx]
                top = y - random.randint(0, 128)
                left = x - random.randint(0, 128)
                top = max(0, top)
                left = max(0, left)
                if top + 256 > 512:
                    top = 512 - 256
                if left + 256 > 640:
                    left = 640 - 256
            [n, t, m_M, n_M] = shape(MixData_out)

            LabelData_Img_1 = LabelData_Img[top:top + 256, left:left + 256]
            LabelData = torch.from_numpy(LabelData_Img_1)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            MixData_out_1 = MixData_out[:, :, top:top + 256, left:left + 256]

            return MixData_out_1, TgtData_out, m_L, n_L

        else:
            raise FileNotFoundError(f"size wong!!!")

    def __len__(self):
        return len(self.imgs_arr)



class CSIG_TestSetLoader(Dataset):
    def __init__(self, root, fullSupervision=False):

        txtpath = root + 'val_list.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.root = root        # dataset/CSIG/
        self.fullSupervision = fullSupervision
        self.train_mean = 105.31
        self.train_std = 26.08

    def extract_number(self, filename):
        return int(os.path.splitext(filename)[0])

    def __getitem__(self, index):
        img_path_mix = self.imgs_arr[index]  # 'dataset/CSIG/train\000001\img\000001.png'

        img_path_mix = img_path_mix.replace('\\', '/')
        dir_path, filename = os.path.split(img_path_mix)  # 'dataset/CSIG/train/000001/img', '000007.png'
        folder_name = os.path.basename(os.path.dirname(dir_path))  # '000001'
        img_dir_root = os.path.dirname(dir_path)  # 'dataset/CSIG/train/000001'

        file_basename = os.path.splitext(filename)[0]
        ext = '.png'
        img_path_tgt = os.path.join(img_dir_root, 'mask', file_basename + ext)
        
        LabelData_Img = Image.open(img_path_tgt)
        LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
        LabelData_Img[LabelData_Img > 0] = 1.0

        img_dir = os.path.join(img_dir_root, 'img')
        img_list = sorted(os.listdir(img_dir), key=self.extract_number)

        try:
            idx = img_list.index(filename)
        except ValueError:
            raise FileNotFoundError(f"{filename} not found in {img_dir}")

        if idx >= 4:
            indices = range(idx - 4, idx + 1)
        else:
            indices = range(idx, idx + 5)
            indices = indices[::-1]

        MixData_Img = []
        for i in indices:
            if i >= len(img_list):
                break
            img_path = os.path.join(img_dir, img_list[i])
            img = Image.open(img_path)
            img_np = np.array(img, dtype=np.float32)
            MixData_Img.append(img_np)

        MixData_Img = np.stack(MixData_Img, axis=0).astype(np.float32)
        MixData_Img = (MixData_Img - self.train_mean) / self.train_std
        MixData = torch.from_numpy(MixData_Img)
        MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)

        [m_L, n_L] = np.shape(LabelData_Img)

        LabelData = torch.from_numpy(LabelData_Img)
        TgtData_out = torch.unsqueeze(LabelData, 0)
        return MixData_out, TgtData_out, m_L, n_L
        # if m_L == 256 and n_L == 256:
        #     # Tgt preprocess
        #     LabelData = torch.from_numpy(LabelData_Img)
        #     TgtData_out = torch.unsqueeze(LabelData, 0)
        #     return MixData_out, TgtData_out, m_L, n_L

        # elif m_L == 1024 and n_L == 1024:
        #     ys, xs = np.where(LabelData_Img > 0)
        #     if len(ys) == 0:
        #         top = random.randint(0, m_L - 256)
        #         left = random.randint(0, n_L - 256)
        #     else:
        #         idx = random.randint(0, len(ys) - 1)
        #         y, x = ys[idx], xs[idx]
        #         top = y - random.randint(0, 128)
        #         left = x - random.randint(0, 128)
        #         top = max(0, top)
        #         left = max(0, left)
        #         if top + 256 > 1024:
        #             top = 1024 - 256
        #         if left + 256 > 1024:
        #             left = 1024 - 256
        #     [n, t, m_M, n_M] = shape(MixData_out)

        #     LabelData_Img_1 = LabelData_Img[top:top + 256, left:left + 256]
        #     LabelData = torch.from_numpy(LabelData_Img_1)
        #     TgtData_out = torch.unsqueeze(LabelData, 0)
        #     MixData_out_1 = MixData_out[:, :, top:top + 256, left:left + 256]

        #     return MixData_out_1, TgtData_out, m_L, n_L

        # elif m_L == 512 and n_L == 640:
        #     ys, xs = np.where(LabelData_Img > 0)
        #     if len(ys) == 0:
        #         top = random.randint(0, m_L - 256)
        #         left = random.randint(0, n_L - 256)
        #     else:
        #         idx = random.randint(0, len(ys) - 1)
        #         y, x = ys[idx], xs[idx]
        #         top = y - random.randint(0, 128)
        #         left = x - random.randint(0, 128)
        #         top = max(0, top)
        #         left = max(0, left)
        #         if top + 256 > 512:
        #             top = 512 - 256
        #         if left + 256 > 640:
        #             left = 640 - 256
        #     [n, t, m_M, n_M] = shape(MixData_out)

        #     LabelData_Img_1 = LabelData_Img[top:top + 256, left:left + 256]
        #     LabelData = torch.from_numpy(LabelData_Img_1)
        #     TgtData_out = torch.unsqueeze(LabelData, 0)
        #     MixData_out_1 = MixData_out[:, :, top:top + 256, left:left + 256]

        #     return MixData_out_1, TgtData_out, m_L, n_L

        # else:
        #     raise FileNotFoundError(f"size wong!!!")


    def __len__(self):
        return len(self.imgs_arr)
