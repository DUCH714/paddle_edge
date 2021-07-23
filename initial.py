import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage

import paddle
from paddle.nn import functional as F

IMAGE_SIZE = (400, 400)
# train_images_path = "./image/"
# label_images_path = "./edge/"
# image_count = len([os.path.join(train_images_path, image_name)
#                    for image_name in os.listdir(train_images_path)
#                    if image_name.endswith('.png')])
# print("用于训练的图片样本数量:", image_count)
#
#
# # 对数据集进行处理，划分训练集、测试集
# def _sort_images(image_dir, image_type):
#     """
#     对文件夹内的图像进行按照文件名排序
#     """
#     files = []
#
#     for image_name in os.listdir(image_dir):
#         if image_name.endswith('.{}'.format(image_type)) \
#                 and not image_name.startswith('.'):
#             files.append(os.path.join(image_dir, image_name))
#
#     return sorted(files)
#
#
# def write_file(mode, images, labels):
#     with open('./{}.txt'.format(mode), 'w') as f:
#         for i in range(len(images)):
#             f.write('{}\t{}\n'.format(images[i], labels[i]))
#
#
# """
# 由于所有文件都是散落在文件夹中，在训练时需要使用的是数据集和标签对应的数据关系，
# 所以第一步是对原始的数据集进行整理，得到数据集和标签两个数组，分别一一对应。
# 这样可以在使用的时候能够很方便的找到原始数据和标签的对应关系，否则对于原有的文件夹图片数据无法直接应用。
# 在这里是用了一个非常简单的方法，按照文件名称进行排序。
# 因为刚好数据和标签的文件名是按照这个逻辑制作的，名字都一样，只有扩展名不一样。
# """
# images = _sort_images(train_images_path, 'png')
# labels = _sort_images(label_images_path, 'png')
# eval_num = int(image_count * 0.15)
# print(len(labels))
# write_file('test', images[0:eval_num], labels[0:eval_num])
# write_file('train', images[eval_num:], labels[eval_num:])
# write_file('predict', images[-eval_num:], labels[-eval_num:])

##########################################################################
import random

from paddle.io import Dataset
from paddle.vision.transforms import transforms as T


class PetDataset(Dataset):
    """
    数据集定义
    """

    def __init__(self, mode='train'):
        """
        构造函数
        """
        self.image_size = IMAGE_SIZE
        self.mode = mode.lower()

        assert self.mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(self.mode)

        self.train_images = []
        self.label_images = []

        with open('./{}.txt'.format(self.mode), 'r') as f:
            for line in f.readlines():
                image, label = line.strip().split('\t')
                self.train_images.append(image)
                self.label_images.append(label)

    def _load_img(self, path, color_mode='rgb', transforms=[]):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        with open(path, 'rb') as f:
            img = PilImage.open(io.BytesIO(f.read()))
            if color_mode == 'grayscale':
                # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
                # convert it to an 8-bit grayscale image.
                if img.mode not in ('L', 'I;16', 'I'):
                    img = img.convert('L')
            elif color_mode == 'rgba':
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif color_mode == 'rgb':
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')

            return T.Compose( transforms)(img)

    def __getitem__(self, idx):
        """
        返回 image, label
        """
        train_image = self._load_img(self.train_images[idx],
                                     transforms=[T.Transpose()])  # 加载原始图像
        label_image = self._load_img(self.label_images[idx],
                                     transforms=[T.Transpose()])  # 加载Label图像

        # 返回image, label
        train_image = np.array(train_image, dtype='float32')
        label_image = np.array(label_image, dtype='float32')
        return train_image, label_image

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.train_images)
#####################################################################################

import paddle.nn as nn
from paddle.nn.initializer import Assign
#############################################################################
w = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='float32')
w = w.reshape([1, 1, 3, 3])
# 由于输入通道数是3，将卷积核的形状从[1,1,3,3]调整为[1,3,3,3]
w = np.repeat(w, 3, axis=1)
w=np.repeat(w,3,axis=0)
# 创建卷积算子，输出通道数为3，卷积核大小为3x3，
# 并使用上面的设置好的数值作为卷积核权重的初始化参数
network = nn.Sequential(nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[3, 3],
              weight_attr=paddle.ParamAttr(
                  initializer=Assign(value=w))),nn.Pad2D(padding=[1,1,1,1], mode='constant'))

model = paddle.Model(network)
train_dataset = PetDataset(mode='train') # 训练数据集
val_dataset = PetDataset(mode='test') # 验证数据集

optim = paddle.optimizer.Adam(parameters=model.parameters())
model.prepare(optim, paddle.nn.L1Loss(reduction='mean'))
model.fit(train_dataset,
          val_dataset,
          epochs=1000,
          batch_size=32,
          verbose=1)
params_info  = model.parameters()
print(params_info)
