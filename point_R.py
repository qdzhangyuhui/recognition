import cv2
import numpy as np
import os


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def point_R(path):
    img_original = cv2.imread(path)
    shape = img_original.shape
    img = cv2.resize(img_original, (int(shape[1] * 80 / shape[0]), 80))  # 统一图片高度
    value_of_gamma = 180  # gamma取值
    value_of_gamma = value_of_gamma * 0.01  # 压缩gamma范围，以进行精细调整
    image_gamma_correct = gamma_trans(img, value_of_gamma)  # 2.5为gamma函数的指数值，大于1曝光度下降，大于0小于1曝光度增强
    img_Gaussi = cv2.GaussianBlur(image_gamma_correct, (9, 9), 9)  # 高斯滤波 ( Gaussian Filtering )
    gray = cv2.cvtColor(img_Gaussi, cv2.COLOR_BGR2GRAY)  # 灰度化
    ret, image_deal = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 二值化
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(image_deal, kernel1)  # 膨胀
    # cv2.imshow('image_dilate', dilate)
    eroded = cv2.erode(dilate, kernel2)  # 腐蚀
    image, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prediction = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < len(image) * len(image[0]) * 0.001:
            continue
        if area < len(image) * len(image[0]) * 0.01:
            return -2
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
        cv2.drawContours(image, [box], -1, (255, 255, 255), 1)
        box = sorted(box, key=lambda CD: CD[1])
        t_points = box[:2]
        t_points = sorted(t_points, key=lambda CD: CD[0])
        b_points = box[2:]
        b_points = sorted(b_points, key=lambda CD: CD[0])
        if t_points[0][0] != b_points[0][0]:
            K = (t_points[0][1] - b_points[0][1]) / (t_points[0][0] - b_points[0][0])
        else:
            K = 0
        if K >= 0:
            w = abs(b_points[1][0] - b_points[0][0])
            h = abs(b_points[1][1] - t_points[1][1])
            count = 0
            if b_points[1][1] > len(image):
                b_points[1][1] = len(image)
            if b_points[1][0] > len(image[0]):
                b_points[1][0] = len(image[0])
            # print(len(image), len(image[0]), b_points[1][1], b_points[1][0])
            for i in range(int(b_points[1][1] - 0.15 * h), b_points[1][1]):
                for j in range(int(b_points[1][0] - 1 / 3 * w), b_points[1][0]):
                    if image[i][j] == 255:
                        count += 1
            ratio = count / ((int(b_points[1][1] - 1 / 4 * h) - b_points[1][1]) * (
                    int(b_points[1][0] - 1 / 3 * w) - b_points[1][0]))
            prediction.append([b_points[0][0], ratio])
            # print(ratio)
        else:
            prediction.append([b_points[0][0], 0])
        # cv2.imshow('', image)
        # cv2.waitKey(0)
    if len(prediction) == 0:
        return -1
    prediction = sorted(prediction, key=lambda item: item[0])
    for i in range(len(prediction)):
        prediction[i][0] = i
    prediction = sorted(prediction, key=lambda item: -item[1])
    # print(prediction)
    if prediction[0][1] > 0:
        return prediction[0][0]
    else:
        return -1


if __name__ == '__main__':
    path = 'image_all'
    for filename in os.listdir(path):
        sub_path = os.path.join(path, filename)
        print(point_R(sub_path))
        img_original = cv2.imread(sub_path)
        cv2.imshow('', img_original)
        cv2.waitKey(0)
