from gfpgan import GFPGANer
import cv2
import numpy as np
import os

# 全局模型实例（只加载一次）
gfpganer = GFPGANer(
    model_path='GFPGAN1\gfpgan\weights\GFPGANv1.4.pth',
    upscale=1,  # 不放大图像
    arch='clean',  # 模型架构
    channel_multiplier=2,  # 通道乘数
    bg_upsampler=None  # 背景上采样器
)


def beautify_face(img_bgr):
    """
    使用GFPGAN模型美化人脸

    参数:
        img_bgr (np.ndarray): BGR格式的输入图像

    返回:
        np.ndarray: 美化后的BGR图像

    异常:
        ValueError: 输入图像无效
        RuntimeError: GFPGAN处理失败
    """
    # 输入验证
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("输入图片无效")

    # 转换颜色空间 (BGR -> RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        # 使用GFPGAN增强图像
        result = gfpganer.enhance(
            img_rgb,
            has_aligned=False,  # 非对齐人脸
            only_center_face=False,  # 处理所有人脸
            paste_back=True  # 贴回原图
        )

        # 解析结果元组
        if isinstance(result, tuple):
            _, _, restored_img = result  # 提取修复后的图像
        else:
            restored_img = result
    except Exception as e:
        raise RuntimeError(f"GFPGAN推理失败: {e}")

    # 验证输出
    if restored_img is None or not isinstance(restored_img, np.ndarray):
        raise ValueError("GFPGAN未能返回有效图片")

    # 转换回BGR格式
    return cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)