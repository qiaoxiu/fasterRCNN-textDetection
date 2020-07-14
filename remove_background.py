import cv2
import numpy as np
import scipy.signal as signal



def img_slice(src, row=3, col=1):
    _H, _W, _D = src.shape
    h = int(_H / row)
    w = int(_W / col)
    slices = []
    for i in range(row):
        for j in range(col):
            s = src[i * h:(i + 1) * h, j * w:(j + 1) * w]
            slices.append(s)
    return slices


def img_concat(ls, row=3, col=1):
    assert row * col == len(ls), "slices concat error: shape not match"
    v_ = []
    for i in range(row):
        h_ = ls[i * col:(i + 1) * col]
        h_concat = cv2.hconcat(h_)
        v_.append(h_concat)
    v_concat = cv2.vconcat(v_)
    return v_concat


def fg_bg_split(img):
    pic = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img = cv2.erode(img, kernel)
    # cv2.imshow('erosion', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    cl = clahe.apply(gray)
    otsu_val, th = cv2.threshold(cl, 0, 255, cv2.THRESH_OTSU)
    mask = 255 - th
    fg = cv2.bitwise_and(pic, pic, mask=mask)
    bg = cv2.bitwise_and(pic, pic, mask=th)
    return fg, bg


def mean_blur(array):
    n = 20
    neighbor = np.array([(1 / n)] * n)
    new_array = signal.convolve(array, neighbor, mode='same')
    return new_array


def smooth_percent_hist(img):
    img = img.ravel()
    img = img[img > 0]
    img = img[img < 255]
    img_hist = np.bincount(img, minlength=256)
    img_rate_hist = img_hist*1.0 / sum(img_hist)
    smooth_array = mean_blur(img_rate_hist)
    return smooth_array


def compute_cross_point(fg_smooth_array, bg_smooth_array, n=300, tol=0.0005, leftval=80):
    xp = np.arange(0, 256)
    x = np.linspace(0, 255, num=256 * n)
    fg_expan = np.interp(x, xp, fg_smooth_array)
    bg_expan = np.interp(x, xp, bg_smooth_array)
    bg_max_index = np.argmax(bg_expan)
    diff = abs(fg_expan - bg_expan)

    index = np.argwhere(diff < tol)
    index = index[index > leftval * n]
    index = index[index < bg_max_index]
    if len(index) == 0:
        index = [bg_max_index]
    minval_idx = int(max(index) / n)
    return minval_idx


def rgb_cross_point(fg, bg):
    fg_b, fg_g, fg_r = cv2.split(fg)
    bg_b, bg_g, bg_r = cv2.split(bg)
    fg_b_smooth_array = smooth_percent_hist(fg_b)
    bg_b_smooth_array = smooth_percent_hist(bg_b)
    b_minval_idx = compute_cross_point(fg_b_smooth_array, bg_b_smooth_array)

    fg_g_smooth_array = smooth_percent_hist(fg_g)
    bg_g_smooth_array = smooth_percent_hist(bg_g)
    g_minval_idx = compute_cross_point(fg_g_smooth_array, bg_g_smooth_array)

    fg_r_smooth_array = smooth_percent_hist(fg_r)
    bg_r_smooth_array = smooth_percent_hist(bg_r)
    r_minval_idx = compute_cross_point(fg_r_smooth_array, bg_r_smooth_array)
    return b_minval_idx, g_minval_idx, r_minval_idx


def linear_map(minlevel, maxlevel):
    if minlevel > maxlevel:
        print("minlevel > maxlevel")
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):
            if i < minlevel:
                newmap[i] = 0
            elif i > maxlevel:
                newmap[i] = 253
            else:
                newmap[i] = (i - minlevel)*1.0 / (maxlevel - minlevel) * 255
        return newmap


def createnewimg_by_point(img, crosspoint, blackval=40):
    h, w, d = img.shape
    new_img = np.zeros([h, w, d])
    for (i, minval_idx) in zip(np.arange(d), crosspoint):
        newmap = linear_map(blackval, minval_idx)
        if len(newmap) == 0:
            continue
        for j in range(h):
            new_img[j, :, i] = newmap[img[j, :, i]]
    return new_img


def computehist(img):
    hist = np.bincount(img.ravel(), minlength=256)
    return hist


def computeminlevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if sum >= (pnum * rate * 0.01):
            return i


def computemaxlevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255 - i]
        if sum >= (pnum * rate * 0.01):
            return 255 - i


def createnewimg_by_rate(img, delrate=80, blackval=40):
    h, w, d = img.shape
    new_img = np.zeros([h, w, d])
    for i in range(d):
        imghist = computehist(img[:, :, i])
        minlevel = computeminlevel(imghist, 0.01, h * w)
        maxlevel = computemaxlevel(imghist, delrate, h * w)
        # print(minlevel)
        # print(maxlevel)
        newmap = linear_map(blackval, maxlevel)
        if len(newmap) == 0:
            continue
        for j in range(h):
            new_img[j, :, i] = newmap[img[j, :, i]]
    return new_img


def createnewimg_by_bg_rate(img, bg, delrate=90, blackval=40):
    h, w, d = img.shape
    new_img = np.zeros([h, w, d])
    for i in range(d):
        bghist = computehist(bg[:, :, i])
        bghist[0] = 0
        maxlevel = computemaxlevel(bghist, delrate, sum(bghist))
        # print(maxlevel)
        newmap = linear_map(blackval, maxlevel)
        if len(newmap) == 0:
            continue
        for j in range(h):
            new_img[j, :, i] = newmap[img[j, :, i]]
    return new_img


def fg_bg_hsv_hist_show():
    img = cv2.imread(r'aa.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)

    fg, bg = fg_bg_split(img)

    fg_hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    fg_h, fg_s, fg_v = cv2.split(fg_hsv)
    bg_h, bg_s, bg_v = cv2.split(bg_hsv)

    fg_h_smooth_array = smooth_percent_hist(fg_h)
    bg_h_smooth_array = smooth_percent_hist(bg_h)
    # plt.plot(fg_h_smooth_array, color='b')
    # plt.plot(bg_h_smooth_array, color='c')
    # plt.show()

    fg_s_smooth_array = smooth_percent_hist(fg_s)
    bg_s_smooth_array = smooth_percent_hist(bg_s)
    # plt.plot(fg_s_smooth_array, color='g')
    # plt.plot(bg_s_smooth_array, color='y')
    # plt.show()

    fg_v_smooth_array = smooth_percent_hist(fg_v)
    bg_v_smooth_array = smooth_percent_hist(bg_v)
    # plt.plot(fg_v_smooth_array, color='r')
    # plt.plot(bg_v_smooth_array, color='m')
    # plt.show()

    lower_1 = np.array([0, 0, 0])
    upper_1 = np.array([22, 22, 228])
    mask_1 = cv2.inRange(hsv, lower_1, upper_1)

    lower_2 = np.array([44, 83, 0])
    upper_2 = np.array([179, 255, 228])
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)

    lower_3 = np.array([-1, -1, -1])
    upper_3 = np.array([1, 1, 1])
    mask_3 = cv2.inRange(hsv, lower_3, upper_3)

    lower_bg = np.array([20, 22, 228])
    upper_bg = np.array([44, 126, 255])
    mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)

    mask = mask_1 + mask_2 + mask_3

    cv2.imshow('mask_bg', mask_bg)
    cv2.imshow('mask_3', mask_3)
    cv2.imshow('mask', mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    res_fg = cv2.bitwise_and(img, img, mask=255 - mask_bg)
    # res = res + 255
    cv2.imshow('res', res)
    cv2.imshow('res_fg', res_fg)
    cv2.waitKey(0)


# 分离前景与背景，根据前景和背景的曲线交点，进行背景去除
def bguni_by_crosspoint(img, row=6, col=1, offset=0, blackval=40, rounds=2):
    img_ls = img_slice(img, row, col)
    newimg_ls = []
    for im in img_ls:
        for rud in range(rounds):
            fg, bg = fg_bg_split(im)
            crosspoint = rgb_cross_point(fg, bg)
            crosspoint = [val + offset for val in crosspoint]
            im = createnewimg_by_point(im, crosspoint, blackval)
            im = np.array(im, dtype='uint8')
        newimg_ls.append(im)
    concat = img_concat(newimg_ls, row, col)
    return concat


# 分离前景与背景，去除背景90%的高阶色彩
import sys

def bguni_by_bgrate(img, row=6, col=1, offset=0, delrate=90, blackval=40, rounds=1):
    img_ls = img_slice(img, row, col)
    newimg_ls = []
    for im in img_ls:
        for rud in range(rounds):
            fg, bg = fg_bg_split(im)
            im = createnewimg_by_bg_rate(im, bg, delrate, blackval)
            im = np.array(im, dtype='uint8')
        newimg_ls.append(im)
    concat = img_concat(newimg_ls, row, col)
    return concat


# 去除整张图像80%的高阶色彩
def bguni_by_rate(img, row=6, col=1, offset=0, delrate=80, blackval=40, rounds=1):
    img_ls = img_slice(img, row, col)
    newimg_ls = []
    for im in img_ls:
        for rud in range(rounds):
            if rud != 0:
                blackval = blackval - 20 * rud
            im = createnewimg_by_rate(im, delrate, blackval)
            im = np.array(im, dtype='uint8')
        newimg_ls.append(im)
    concat = img_concat(newimg_ls, row, col)
    return concat

import os
def load_img_path(images_path,prefix):
    if os.path.isdir(images_path):
        file_path_list = []
        file_list = os.walk(images_path)
        for i in file_list:
            if len(i[2]) > 0:
                for name in i[2]:
                    if os.path.join(i[0], name).__contains__(prefix):
                        file_path_list.append(os.path.join(i[0], name))
        print(images_path + '  目录下包含：{}  张图片'.format(len(file_path_list)))
        file_path_list.sort()
        assert int(len(file_path_list)) > 0
        # return np.array(file_path_list).reshape(-1, 1), int(len(file_path_list))
        return file_path_list, int(len(file_path_list))


    else:
        return [images_path], len([images_path])

def remove_bg_by_rate(img0):
    return  bguni_by_bgrate(img0, row=1, col=1, offset=0, delrate=90, blackval=1, rounds=1)


if __name__ == '__main__':
    for each_file in load_img_path(r'C:\Users\Desktop\result_imgs', 'JPG')[0]:
        # img = cv2.imread(r'D:\cyclegan\red\A47.jpg')

        img = cv2.imread(each_file)
        bg_uniform = bguni_by_bgrate(img, row=1, col=1, offset=0, delrate=90, blackval=1, rounds=1)
        print(bg_uniform.shape)
        cv2.imwrite(each_file, bg_uniform)



