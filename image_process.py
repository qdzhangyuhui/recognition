import cv2
from PIL import Image
import pytesseract
import numpy as np
import os
from tkinter import filedialog
name = 'gp'
def normalize_figure(data):
    length = len(data)
    figure = ''
    for i in range(length):
        c = data[i]
        if not c.isspace():
            if c.isdigit():
                figure += data[i]
            if c == '.' and i != length-1 and i != 0:
                figure += data[i]
    return figure

def  gamma_trans(img,gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def main():
    page = 1
    path = filedialog.askdirectory()
    with  open('data2.txt', 'w') as doc:
        for lists in os.listdir(path):
            sub_path = os.path.join(path, lists)
            if os.path.isfile(sub_path):
                img_original = cv2.imread(sub_path)
                shape = img_original.shape
                img = cv2.resize(img_original, (int(shape[1] * 80 / shape[0]), 80))  # 统一图片高度
                value_of_gamma = 180  # gamma取值
                value_of_gamma = value_of_gamma * 0.01  # 压缩gamma范围，以进行精细调整
                image_gamma_correct = gamma_trans(img, value_of_gamma)  # 2.5为gamma函数的指数值，大于1曝光度下降，大于0小于1曝光度增强
                # cv2.imshow("image_gamma_correct", image_gamma_correct)
                # blur = cv2.bilateralFilter( image_gamma_correct, 5, 10, 10)  # 双边滤波更注重保持边缘原状
                img_Gaussi = cv2.GaussianBlur(image_gamma_correct, (9, 9), 9)  # 高斯滤波 ( Gaussian Filtering )
                # cv2.imshow('image_GaussianBlur',dst1)
                # dst2 = cv2.blur(dst1, (1, 1))  # 均值化 ( Gaussian Filtering )
                # img_ = cv2.medianBlur(blur, 3)  # 中值滤波能够保留中值细节
                gray = cv2.cvtColor(img_Gaussi, cv2.COLOR_BGR2GRAY)  # 灰度化
                # cv2.imshow('image_gray',gray)
                ret, image_deal = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 二值化
                # cv2.imshow('image_2', image_deal)
                kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                dilate = cv2.dilate(image_deal, kernel1)  # 膨胀
                # cv2.imshow('image_dilate', dilate)
                eroded = cv2.erode(dilate, kernel2)  # 腐蚀
                # cv2.imshow('image_eroded', eroded)
                cv2.imwrite('d2.jpeg', eroded)
                text = pytesseract.image_to_string(Image.open('d2.jpeg'), lang='num12')
                figure = normalize_figure(text)
                print('%d the ocr recognition is %s'%(page,text))
                print('the normalize figure is %s'%figure)
                name = '%05d' % page
                newname = '/' + '%05d' % page + '.jpeg'
                # print(sub_path)
                os.rename(sub_path, path + newname)
                cv2.imwrite('testnew1/' + name + '_' + text + '.jpeg', eroded)
                doc.write(text)
                page += 1
                # cv2.waitKey(0)



if __name__ == '__main__':

    main()