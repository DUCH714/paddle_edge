import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage

import paddle
from paddle.nn import functional as F

IMAGE_SIZE = (400, 400)
train_images_path = "/home/lifutuan/TianchiYu/paddle/edge/"
label_images_path = "/home/lifutuan/TianchiYu/paddle/edge/"
image_count = len([os.path.join(train_images_path, image_name)
                   for image_name in os.listdir(train_images_path)
                   if image_name.endswith('.png')])
print("用于训练的图片样本数量:", image_count)


# 对数据集进行处理，划分训练集、测试集
def _sort_images(image_dir, image_type):
    """
    对文件夹内的图像进行按照文件名排序
    """
    files = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.{}'.format(image_type)) \
                and not image_name.startswith('.'):
            files.append(os.path.join(image_dir, image_name))

    return sorted(files)


def write_file(mode, images, labels):
    with open('/home/lifutuan/TianchiYu/paddle/{}.txt'.format(mode), 'w') as f:
        for i in range(len(images)):
            f.write('{}\t{}\n'.format(images[i], labels[i]))


"""
由于所有文件都是散落在文件夹中，在训练时需要使用的是数据集和标签对应的数据关系，
所以第一步是对原始的数据集进行整理，得到数据集和标签两个数组，分别一一对应。
这样可以在使用的时候能够很方便的找到原始数据和标签的对应关系，否则对于原有的文件夹图片数据无法直接应用。
在这里是用了一个非常简单的方法，按照文件名称进行排序。
因为刚好数据和标签的文件名是按照这个逻辑制作的，名字都一样，只有扩展名不一样。
"""
images = _sort_images(train_images_path, 'png')
labels = _sort_images(label_images_path, 'png')
eval_num = int(image_count * 0.15)
print(len(labels))
write_file('test', images[0:eval_num], labels[0:eval_num])
write_file('train', images[eval_num:], labels[eval_num:])

