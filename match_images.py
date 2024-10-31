import os
import cv2
import h5py
import numpy as np


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    # kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    try:
        matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    except:
        return []
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def compute_homo(img_original, img_target):
    kp1, des1 = sift_kp(img_original)
    kp2, des2 = sift_kp(img_target)

    if des1 is not None and des2 is not None:
        goodMatch = get_good_match(des1, des2)
        # 当筛选项的匹配对大于4对时：计算视角变换矩阵
        if len(goodMatch) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 4
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
            #  该函数的作用就是先用RANSAC选择最优的四组配对点，再计算H矩阵。H为3*3矩阵
            if H is not None:
                return H

    H = np.eye(3).astype(np.float32)
    return H

def matching(his_img, img):
    H = compute_homo(his_img, img)
    his_img_match = cv2.warpPerspective(his_img, H, (his_img.shape[1], his_img.shape[0]))
    return his_img_match


# if __name__ == '__main__':
#     his_img = cv2.imread('./0.bmp')
#     img = cv2.imread('./20.bmp')
#     his_img_match = matching(his_img, img)
#     lrj = 1
