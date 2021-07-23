import cv2
import matplotlib.pyplot as plt
import os
for filename in os.listdir('/home/lifutuan/TianchiYu/DIV2K/DIV2K_train_HR'):
    img=cv2.imread('/home/lifutuan/TianchiYu/DIV2K/DIV2K_train_HR/'+filename)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('/home/lifutuan/TianchiYu/paddle/image/'+filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    #plt.show()`

