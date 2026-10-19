# -*- coding: utf-8 -*-
# @Time    : 2/3/26 7:28 PM
# @File    : load_data.py
# @Software: VScode

import sys
sys.path.append('..')

import re
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
import numpy as np

dataset_id = 1
# ============= dataset1, 人脸 ============
if dataset_id == 0: # 眼底图像
    data_root = '/home/upload/Tongue_Label/ODIR-6k/'
    label_root= '/home/upload/Tongue_Label/ODIR-6k/'
    # 更新标签20260611
    pos_label_path = os.path.join(label_root, 'ODIR6k_Label_all.txt')
elif dataset_id == 1:
# ============== dataset2, 舌象 ==============
    data_root = '/home/upload/'
    label_root= '/home/upload/Tongue_Label/'
    # 更新标签20240428
    dis_label_path = os.path.join(label_root, 'output_label.txt')


ds_id = 1  # 舌象特征分类1; 舌象特征分类2；
if ds_id == 0:
    # 每个通道的均值： [0.1554478  0.2814865  0.43699695]
    # 每个通道的标准差： [0.1469835  0.20919825 0.28942842]

    # 每个通道的均值： [0.43966475, 0.47025456, 0.4956167 ]
    # 每个通道的标准差： [0.26568854, 0.27167103, 0.26902942]
    data_dir = os.path.join(data_root,'preprocessed_images')   # tonguePhoto_dataset0,1,2,3,4
    data_mean, data_std = [0.1554478, 0.2814865, 0.43699695], [0.1469835, 0.20919825, 0.28942842]  # tongue
elif ds_id == 1:
    # 每个通道的均值：  [0.4914228,0.4994282,0.73754448]
    # 每个通道的标准差：[0.17772971,0.1571762,0.17106681]
    data_dir = os.path.join(data_root,'Tonguedataset2') # tonguePhoto_dataset0,1,2,3,4
    data_mean, data_std = [0.4914228,0.4994282,0.73754448], [0.17772971,0.1571762,0.17106681]  # tongue


# ===== 加载舌象数据 ======
# class TongueDiseasePosition(Dataset):
#     '''
#     制作自己的数据集，可以记录每个样本。
#     '''
#     def __init__(self, root, flag=None, transform=None):
#         self._root = os.path.expanduser(root)
#         self._flag = flag
#         self._transform = transform
#         self._exts = ['.jpg', '.jpeg', '.png']
#         self.cid_labels = {}

#         self._get_cid_labels_sym()
#         self.items = []
#         if self._flag == 1:
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder1'+ '/' + 'train.txt'))
#             self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder2'+ '/' + 'train.txt'))   #0428完成
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder3'+ '/' + 'train.txt'))
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder4'+ '/' + 'train.txt'))
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder5'+ '/' + 'train.txt'))
#         else:
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder1' + '/' + 'validation.txt'))
#             self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder2' + '/' + 'validation.txt')) #0428完成
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder3' + '/' + 'validation.txt'))
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder4' + '/' + 'validation.txt'))
#             # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder5' + '/' + 'validation.txt'))

#     def _get_cid_labels_sym(self):
#         flag = 1
#         for line in open(pos_label_path):
#             if flag:
#                 flag = 0
#                 continue
#             line_split = line.strip('\n').replace(',', '').split()
#             line_cid = int(line_split[0])
#             line_label_str_ls = line_split[1:]
#             line_label = [int(i) for i in line_label_str_ls]
#             self.cid_labels[line_cid] = line_label

#     def _getlabel(self, filename):
#         cid = int(filename.split('.')[0][3:])
#         label = self.cid_labels[cid]
#         # print(label.shape)

#         return label
    
#     def _list_images(self, root, filenames_path):  # filenames_path 为train.txt的路径 或valid
#         self.items = []
#         print('loading TongueDiseasePosition . . .')
#         with open(filenames_path, 'r') as f:
#             filenames = f.readlines()
#             for filename in filenames:
#                 # filename = filename.strip()
#                 filename = filename.strip('\n')
#                 # print(filename)
#                 try:
#                     label = self._getlabel(filename)
#                     # print(label)
#                     self.items.append((filename, label))  # filename是图片名
#                 except Exception as e:
#                     print('Error when loading ', filename)
#                     continue

#     def __getitem__(self, idx):
#         img = Image.open(self._root + '/' + self.items[idx][0])
#         label = self.items[idx][1]
#         if self._transform is not None:
#             img = self._transform(img)
#         label = torch.FloatTensor(label)    # 增加图片的下标

#         return img, label

#     def __len__(self):
#         return len(self.items)

# ===== 提取文件名的数字ID（支持 数字.jpg, 数字_left.jpg, 数字_right.jpg）=====
def extract_id_from_filename(filename):
    basename = os.path.splitext(filename)[0]
    match = re.match(r'(\d+)', basename)      # 开头数字
    if match is None:
        match = re.search(r'\d+', basename)   # 任意位置第一组数字
    if match is None:
        raise ValueError(f"无法从文件名 '{filename}' 中提取数字ID")
    return int(match.group())

# ===== 加载舌象数据（位置标签）=====
class TongueDiseasePosition(Dataset):
    def __init__(self, root, flag=None, transform=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self.cid_labels = {}
        self._get_cid_labels_sym()
        self.items = []
        if self._flag == 1:
            train_txt = os.path.join(self._root, 'DNDL_five_folder/folder2/train.txt')
            self._list_images(train_txt)
        else:
            val_txt = os.path.join(self._root, 'DNDL_five_folder/folder2/validation.txt')
            self._list_images(val_txt)

    def _get_cid_labels_sym(self):
        flag = 1
        for line in open(pos_label_path):
            if flag:
                flag = 0
                continue
            parts = line.strip('\n').replace(',', '').split()
            cid = int(parts[0])
            label = [int(i) for i in parts[1:]]
            self.cid_labels[cid] = label

    def _getlabel(self, filename):
        cid = extract_id_from_filename(filename)
        if cid not in self.cid_labels:
            raise KeyError(f"ID {cid} 不在标签文件中")
        return self.cid_labels[cid]

    def _list_images(self, txt_path):
        self.items = []
        print('loading TongueDiseasePosition from', txt_path)
        with open(txt_path, 'r') as f:
            for line in f:
                filename = line.strip()
                try:
                    label = self._getlabel(filename)
                    self.items.append((filename, label))
                except Exception as e:
                    print(f'跳过 {filename}: {e}')

    def __getitem__(self, idx):
        img_path = os.path.join(self._root, self.items[idx][0])
        img = Image.open(img_path)
        label = self.items[idx][1]
        if self._transform:
            img = self._transform(img)
        return img, torch.FloatTensor(label)

    def __len__(self):
        return len(self.items)


class TongueDiseaseSymptom(Dataset):
    '''
    制作自己的数据集，可以记录每个样本。
    '''
    def __init__(self, root, flag=None, transform=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self.cid_labels = {}

        self._get_cid_labels_sym()
        self.items = []
        if self._flag == 1:
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder1'+ '/' + 'train.txt'))
            self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder2'+ '/' + 'train.txt'))   #0428完成
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder3'+ '/' + 'train.txt'))
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder4'+ '/' + 'train.txt'))
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder5'+ '/' + 'train.txt'))
        else:
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder1' + '/' + 'validation.txt'))
            self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder2' + '/' + 'validation.txt')) #0428完成
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder3' + '/' + 'validation.txt'))
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder4' + '/' + 'validation.txt'))
            # self._list_images(os.path.join(self._root), os.path.join(self._root, 'DNDL_five_folder'+ '/' + 'folder5' + '/' + 'validation.txt'))

    def _get_cid_labels_sym(self):
        flag = 1
        for line in open(dis_label_path):
            if flag:
                flag = 0
                continue
            line_split = line.strip('\n').replace(',', '').split()
            line_cid = int(line_split[0])
            line_label_str_ls = line_split[1:]
            line_label = [int(i) for i in line_label_str_ls]
            self.cid_labels[line_cid] = line_label

    def _getlabel(self, filename):
        cid = int(filename.split('.')[0][3:])
        label = self.cid_labels[cid]
        # print(label.shape)

        return label
    
    def _list_images(self, root, filenames_path):  # filenames_path 为train.txt的路径 或valid
        self.items = []
        print('loading TongueDiseaseSymptom . . .')
        with open(filenames_path, 'r') as f:
            filenames = f.readlines()
            for filename in filenames:
                # filename = filename.strip()
                filename = filename.strip('\n')
                # print(filename)
                try:
                    label = self._getlabel(filename)
                    # print(label)
                    self.items.append((filename, label))  # filename是图片名
                except Exception as e:
                    print('Error when loading ', filename)
                    continue

    def __getitem__(self, idx):
        img = Image.open(self._root + '/' + self.items[idx][0])
        label = self.items[idx][1]
        if self._transform is not None:
            img = self._transform(img)
        label = torch.FloatTensor(label)    # 增加图片的下标

        return img, label

    def __len__(self):
        return len(self.items)

# =============== load tongue_DL dataset ===============
def get_TongueDiseasePosition(id,transform_train=None, transform_test=None,dataset_id=1):

    # normalize = transforms.Normalize(data_mean, data_std)
    # 每个通道的均值： [0.1554478, 0.2814865, 0.43699695]
    # 每个通道的标准差： [0.1469835, 0.20919825, 0.28942842]
    normalize = transforms.Normalize(mean=[0.1554478, 0.2814865, 0.43699695],
                                           std=[0.1469835, 0.20919825, 0.28942842])

    transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            normalize
            ])

    transform_test = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            normalize
            ])
    
    # data_dir, data_mean, data_std = dir_load(id, dataset_id)
    train_ds = TongueDiseasePosition(os.path.join(data_dir), flag = 1, transform=transform_train)
    test_ds = TongueDiseasePosition(os.path.join(data_dir), flag = 2, transform=transform_test)
    print('==== Load Ton_DL Finish ====')
    return train_ds, test_ds

# =============== load tongue dataset2 ===============
def get_TongueDiseaseSymptom(id,transform_train=None, transform_test=None,dataset_id=1):
    
    # normalize = transforms.Normalize(data_mean, data_std)
    # 每个通道的均值： [0.4914228,0.4994282,0.73754448]
    # 每个通道的标准差： [0.17772971,0.1571762,0.17106681]
    normalize = transforms.Normalize(mean=[0.4914228,0.4994282,0.73754448],
                                           std=[0.17772971,0.1571762,0.17106681])

    transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            normalize
            ])

    transform_test = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            normalize
            ])
    
    # data_dir, data_mean, data_std = dir_load(id, dataset_id)
    train_ds = TongueDiseaseSymptom(os.path.join(data_dir), flag = 1,transform=transform_train)
    test_ds = TongueDiseaseSymptom(os.path.join(data_dir), flag = 2, transform=transform_test)
    print('====Load Ton_DN Finish====')
    return train_ds, test_ds
