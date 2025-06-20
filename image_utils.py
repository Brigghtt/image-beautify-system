import cv2
import numpy as np


def cartoonize_face(img):
    """
    将人脸卡通化

    参数:
        img (np.ndarray): 输入的人脸图像 (BGR)

    返回:
        np.ndarray: 卡通化后的图像 (BGR)
    """
    # 双边滤波保留边缘
    img_color = cv2.bilateralFilter(img, 15, 75, 75)

    # 灰度化和中值滤波
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # 自适应阈值生成边缘
    edges = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 2
    )

    # 合并颜色和边缘
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(img_color_rgb, edges)

    # 转回BGR格式
    return cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)


def whiten_skin(img, y_value=25, blur_ksize=3):
    """
    美白肤色

    参数:
        img (np.ndarray): 输入的人脸图像 (BGR)
        y_value (int): Y通道亮度增加值
        blur_ksize (int): 模糊核大小

    返回:
        np.ndarray: 美白后的图像
    """
    img = img.copy()
    # 转换到YCrCb颜色空间
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 增加亮度通道
    y = cv2.add(y, y_value)

    # 合并通道并转回BGR
    ycrcb = cv2.merge((y, cr, cb))
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 应用高斯模糊
    return cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)


def retro_face(img):
    """
    应用复古效果

    参数:
        img (np.ndarray): 输入的人脸图像 (BGR)

    返回:
        np.ndarray: 复古风格图像
    """
    # 转换到HSV空间并调整饱和度和亮度
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.4  # 降低饱和度
    hsv[..., 2] *= 1.15  # 增加亮度
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 应用LUT调整红色通道
    lut = np.arange(0, 256, dtype=np.uint8)
    lookuptable = np.clip(lut * 0.9 + 30, 0, 255).astype(np.uint8)
    bgr[:, :, 2] = cv2.LUT(bgr[:, :, 2], lookuptable)

    # 调整对比度和亮度
    bgr = cv2.convertScaleAbs(bgr, alpha=1.1, beta=-10)

    # 添加胶片颗粒噪声
    noise = np.random.normal(0, 9, bgr.shape).astype(np.int16)
    grain = cv2.add(bgr.astype(np.int16), noise)
    return np.clip(grain, 0, 255).astype(np.uint8)