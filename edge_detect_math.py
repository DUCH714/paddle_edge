import cv2
from scipy.fft import fft2,ifft2
from PIL import Image
import numpy as np
import copy
import os
import paddle
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign
def laplace(image,Laplace):#[0,1]
    _,_,h=image.shape
    edge=copy.deepcopy(image)
    for i in range(0,h):
        edge[:,:,i]=multiply_fft(image[:,:,i],Laplace)
    return edge # [0,1]
def multiply_fft(U,V):
    AA = (ifft2(fft2(U)*V)).real
    return AA
def construct_V_fft(U):
    n1,n2,_=U.shape
    d1h = np.zeros((n1, n2))
    d1h[0, 0] = -1
    d1h[n1-1, 0] = 1
    d1h = fft2(d1h)
    d2h = np.zeros((n1, n2))
    d2h[0, 0] = -1
    d2h[0, n2-1] = 1
    d2h = fft2(d2h)
    Laplace = abs(d1h)**2 + abs(d2h)**2
    return Laplace
for filename in os.listdir('/home/lifutuan/TianchiYu/DIV2K/DIV2K_train_HR'):
    img=cv2.imread('/home/lifutuan/TianchiYu/DIV2K/DIV2K_train_HR/'+filename)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    img/=255
    Laplace = construct_V_fft(img)
    edge = laplace(img, Laplace)  # get the edge (:,:,3) output:[0,1]
    edge = np.clip(edge, 0, 1)
    edge*=255
    edge = edge.astype(np.uint8)
    cv2.imwrite('/home/lifutuan/TianchiYu/paddle/edge/'+filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    #plt.show()`

