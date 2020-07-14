
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import shutil, os
import cv2
import random
import glob
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola, threshold_local)
import sys

'''
threshold_otsu, 
threshold_niblack,
threshold_sauvola, 
threshold_local
'''
# % matplotlib inline
# plt.figure(figsize=(16, 16))
# fig = plt.figure()
'''
Use to basic image augmentation and copy xml to dir
'''
# 原始数据存储位置
jpg_dir = r'D:\all'
xml_dir=r'C:D:\all'
# 扩充后的数据存储位置
augment_img_dir = r'D:\all\enhance_img'



# ====================================================
# 局部均值亮度的图像二值化
def sauvola_thresh(img_in):
    img_tmp = img_in.copy()
    # window_size = 25
    window_size = 55
    thresh_sauvola = threshold_sauvola(img_tmp, window_size=window_size)

    binary_sauvola = img_tmp > thresh_sauvola
    binary_sauvola = binary_sauvola.astype(np.uint8)
    mask = binary_sauvola == 1
    binary_sauvola[mask] = 255

    return binary_sauvola


# ====================================================
# threshold_local img must be 2-D
def adapt_thresh(img_in):
    img_tmp = img_in.copy()
    block_size = 55  # 35
    adaptive_thresh = threshold_local(img_tmp, block_size, offset=10)
    binary_adaptive = img_tmp > adaptive_thresh
    binary_adaptive = binary_adaptive.astype(np.uint8)
    mask = binary_adaptive == 1  # change to 0/1 from Boole
    binary_adaptive[mask] = 255  # change
    return binary_adaptive


# ==========================================
# 改变亮度
def rnd_bright(img_in):
    img_hsv = img_in.copy()
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = random.uniform(0.8, 1.0) * img_hsv[:, :, 2]
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_hsv


# =================================================================
def rnd_rotate(img_in):  # random rotate
    h, w, _ = img_in.shape
    angle = random.uniform(-1.0, 1.0)  # here define angle range
    M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img_in, M_rotate, (w, h))
    return img_rotated


# =================================================================
def rnd_noise(img_in, ratio=0.05):  # random add noise
    # ratio: scale of how many noise points to add,  h*w*ratio
    h, w, c = img_in.shape
    img_noise = img_in.copy()
    steps = int(h * w * ratio)
    for i in range(0, steps):
        rnd_x = random.randint(1, w - 1)
        rnd_y = random.randint(1, h - 1)
        # rnd_color = random.randint(0,255) #just white points
        if c == 1: img_noise[rnd_y:rnd_x] = random.randint(0, 255)
        if c == 3:
            img_noise[rnd_y, rnd_x, 0] = random.randint(0, 255)
            img_noise[rnd_y, rnd_x, 1] = random.randint(0, 255)
            img_noise[rnd_y, rnd_x, 2] = random.randint(0, 255)
    return img_noise


# =================================================================
def sub_channel(img_in, c='r'):  # use one channel
    sub_img = img_in.copy()
    r = sub_img[:, :, 0]
    g = sub_img[:, :, 1]
    b = sub_img[:, :, 2]
    if c == 'r': channel = r
    if c == 'g': channel = g
    if c == 'b': channel = b
    sub_img = cv2.merge((channel, channel, channel))
    return sub_img


# =================================================================
# random augment input image
def img_augmentation(img_in):
    h, w, _ = img_in.shape

    # random scale 0.7~1.2
    # h_scale = int(h_bg * 0.2 * random.uniform(0.7,1.2)) # 1/(1.2*4rows) = 0.2 not overlap the max height of bg img
    # w_scale = int(h_scale * w/h)
    # piece = cv2.resize(img_in,(w_scale,h_scale))

    # random bright
    img_in = rnd_bright(img_in)
    # random noise
    img_in = rnd_noise(img_in)
    # use one channle
    img_in = sub_channel(img_in, 'r')  # use R channel to remove chop
    # random rotate
    img_in = rnd_rotate(img_in)
    img_out = img_in
    return img_out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# random add line。
def add_line(img):
    image_h = img.shape[0]
    image_w = img.shape[1]
    random_linetype = random.randint(1, 2)  # random choose add linetype。
    mean_value = np.mean(img[:, :, 0])
    color = int(mean_value * random.uniform(0.2, 0.6))
    # print(random_linetype)
    if random_linetype == 1:
        random_h = random.choice(
            [random.randint(1, int(image_h / 5)), random.randint(int(4 * image_h / 5), image_h - 1)])
        pt1 = (0, random_h)
        pt2 = (image_w, random_h)
        img_out = cv2.line(img, pt1, pt2, (color, color, color), thickness=random.choice([1, ]))
    else:
        random_w = random.randint(1, image_w - 1)
        pt3 = (random_w, 0)
        pt4 = (random_w, image_h)
        img_out = cv2.line(img, pt3, pt4, (color, color, color), thickness=random.choice([1, ]))
    return img_out


# 随机参数定义旋转的类型，是顺时针还是逆时针旋转。
def rotate_image(img):
    random_rotatetype = random.randint(1, 2)
    image_h = img.shape[0]
    image_w = img.shape[1]
    if random_rotatetype == 1:
        degree_1 = random.randint(1, 4)
        m_1 = cv2.getRotationMatrix2D((int(image_w / 2), int(image_h / 2)), degree_1, 1)
        img_out = cv2.warpAffine(img, m_1, (int(image_w), int(image_h)), borderMode=cv2.BORDER_REFLECT)
    else:
        degree_2 = random.randint(-4, -1)
        m_2 = cv2.getRotationMatrix2D((int(image_w / 2), int(image_h / 2)), degree_2, 1)
        img_out = cv2.warpAffine(img, m_2, (int(image_w), int(image_h)), borderMode=cv2.BORDER_REFLECT)
    return img_out


# 加盐噪声
def salt_pepper(img, n=0.005, k=0.2):
    m = int(img.shape[0] * img.shape[1] * n)
    for b in range(m):
        i = int(np.random.random() * img.shape[1] - 1)
        j = int(np.random.random() * img.shape[0] - 1)
        a = img[j, i]
        if img.ndim == 2:
            img[j, i] = a * m
            img[j + random.randint(-1, 1), i + random.randint(-1, 1)] = a * k
            img[j + random.randint(-1, 1), i + random.randint(-1, 1)] = a * k
        elif img.ndim == 3:
            img[j, i, 0] = a[0] * m
            img[j + random.randint(-1, 1), i + random.randint(-1, 1), 0] = a[0] * k
            img[j + random.randint(-1, 1), i + random.randint(-1, 1), 0] = a[0] * k
            img[j, i, 1] = a[0] * m
            img[j + random.randint(-1, 1), i + random.randint(-1, 1), 1] = a[0] * k
            img[j + random.randint(-1, 1), i + random.randint(-1, 1), 1] = a[0] * k
            img[j, i, 2] = a[0] * m
            img[j + random.randint(-1, 1), i + random.randint(-1, 1), 2] = a[0] * k
            img[j + random.randint(-1, 1), i + random.randint(-1, 1), 2] = a[0] * k
    return img


# 自定义加黑
def customblur(img):  # enhance img to black
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(img, -1, kernel)
    return dst

def generte_data_main(jpg_dir,xml_dir,augment_img_dir,augment_xml_dir,number):
    """
    :param jpg_dir: 需要扩充的原始图片目录位置
    :param xml_dir: 需要扩充原始图片对应xml文件的目录位置
    :param augment_img_dir: 扩充后图片的存储目录
    :param augment_xml_dir: 扩充后xml文件的存储目录
    :number 对每张图片扩充多少倍
    :return:
    """
    aug_img=''
    #TODO：在后期的数据生成时全部换成透明的贴图方式
    file_list = glob.iglob(jpg_dir + '/*.JPG')  # 遍历()中的指定格式的文件
    for k, img_f in enumerate(file_list):
        img_f = img_f.replace('\\', '/')
        if k > -1:  # 指定从第几张图片开始扩充
            # read image
            img = cv2.imread(img_f)
            img_2 = cv2.imread(img_f, 0)

            print("%i: processing %s" % (k, img_f))
            for i in range(0, int(number)):  # 每张图片扩充100倍，每一倍随机选择一种扩充方式
                xml_f = xml_dir + '/' + os.path.basename(img_f).replace('.jpg', '.xml')
                aug_jpg_name = os.path.basename(img_f).split('.')[0] + '-aug-%i.jpg' % i
                aug_xml_name = os.path.basename(img_f).split('.')[0] + '-aug-%i.xml' % i
                print(aug_jpg_name)
                print(aug_xml_name)
                flag = random.randint(1, 16)  # choose augment method,[1,7];np.random.randint(1,7):[1,7)
                # detection 使用旋转数据集那么原来的Box不会转换到旋转后的位置，不用这个
                # if flag == 11: aug_img = rotate_image(img)
                print(flag)
                # if flag == 10: aug_img = sauvola_thresh(img)
                if flag == 7: aug_img = salt_pepper(img)
                if flag == 9: aug_img = customblur(img)

                if flag == 10: aug_img = add_line(img)
                if flag == 11: aug_img = add_line(img)
                if flag == 12: aug_img = add_line(img)
                if flag == 13: aug_img = add_line(img)
                if flag == 14: aug_img = add_line(img)
                if flag == 15: aug_img = add_line(img)

                # if flag == 6: aug_img = adapt_thresh(img_2)
                if flag == 1: aug_img = rnd_bright(img)
                if flag == 2: aug_img = rnd_rotate(img)
                if flag == 3: aug_img = rnd_noise(img)
                # if flag == 4: aug_img = sub_channel(img)
                if flag == 5: aug_img = img_augmentation(img)
                # aug_img = img_augmentation(img)

                cv2.imwrite(augment_img_dir + '/' + aug_jpg_name, aug_img)
                # shutil.copyfile(xml_f, augment_xml_dir + '/' + aug_xml_name)
                # plt.imshow(aug_img)
                # plt.show()
                # sys.exit(0)
                # break
    print('Completed..............................')

# =================================================================
if __name__ == '__main__':
    file_list = glob.iglob(jpg_dir + '/*.jpg')  # 遍历()中的指定格式的文件
    for k, img_f in enumerate(file_list):
        img_f = img_f.replace('\\', '/')
        if k > -1:  # 指定从第几张图片开始扩充
            # read image
            img = cv2.imread(img_f)
            img_2 = cv2.imread(img_f, 0)

            print("%i: processing %s" % (k, img_f))
            for i in range(0, 9):  # 每张图片扩充n倍，每一倍随机选择一种扩充方式
                if 'JPG' in img_f:
                    xml_f = img_f.replace('.JPG', '.xml')
                else:
                    xml_f = img_f.replace('.jpg', '.xml')
                # txt_f = txt_dir + '/' + os.path.basename(img_f).replace('.jpg', '.txt')
                aug_jpg_name = os.path.basename(img_f).split('.')[0] + '-aug-%i.jpg' % i
                aug_xml_name = os.path.basename(img_f).split('.')[0] + '-aug-%i.xml' % i
                print(img_f)
                print(xml_f)

                # aug_txt_name = os.path.basename(img_f).split('.')[0] + '-aug-%i.txt' % i
                print(aug_jpg_name)
                print(aug_xml_name)
                flag = random.randint(0, 8)  # choose augment method,[1,7];np.random.randint(1,7):[1,7)
                # detection 使用旋转数据集那么原来的Box不会转换到旋转后的位置，不用这个
                # if flag == 11: aug_img = rotate_image(img)
                #TODO 数据扩充应该加强横线竖线的扩充
                print(flag)


                if flag == 0: aug_img = customblur(img)


                if flag == 1: aug_img = rnd_bright(img)
                if flag == 2: aug_img = rnd_rotate(img)
                if flag == 3: aug_img = rnd_noise(img)
                if flag == 4: aug_img = sub_channel(img)
                if flag == 5: aug_img = img_augmentation(img)
                if flag == 6: aug_img = adapt_thresh(img_2)
                if flag == 7: aug_img = salt_pepper(img)
                if flag == 8: aug_img = add_line(img)
                # if flag == 9: aug_img = sauvola_thresh(img)
                # aug_img = img_augmentation(img)

                cv2.imwrite(augment_img_dir + '\\' + aug_jpg_name, aug_img)
                shutil.copyfile(xml_f, augment_img_dir + '\\' + aug_xml_name)

                # shutil.copyfile(xml_f, augment_xml_dir + '/' + aug_xml_name)
    print('Completed..............................')
