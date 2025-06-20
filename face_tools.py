import numpy as np
import cv2
import face_recognition


def is_valid_face_roi(face_roi, min_size=32):
    """
    检查人脸ROI区域是否有效

    参数:
        face_roi (np.ndarray): 人脸区域图像
        min_size (int): 最小有效尺寸（默认32像素）

    返回:
        bool: 是否有效的ROI区域
    """
    # 检查输入是否为有效的numpy数组
    if face_roi is None or not isinstance(face_roi, np.ndarray):
        return False
    # 检查图像是否为空
    if face_roi.size == 0:
        return False
    # 检查图像是否为全黑
    if np.max(face_roi) == 0:
        return False
    # 检查图像尺寸是否过小
    if min(face_roi.shape[:2]) < min_size:
        return False
    return True


def detect_faces_with_landmark(image_bgr):
    """
    检测人脸并提取关键点、生成掩码和分割图像

    参数:
        image_bgr (np.ndarray): BGR格式的输入图像

    返回:
        tuple: (
            face_polys: 人脸多边形轮廓点列表,
            face_masks: 人脸掩码列表,
            face_cuts: 分割出的人脸图像列表,
            img_vis: 带可视化效果的原图
        )
    """
    # 转换颜色空间 (BGR -> RGB)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 人脸检测
    face_locations = face_recognition.face_locations(rgb)
    # 人脸关键点检测
    face_landmarks_list = face_recognition.face_landmarks(
        rgb, face_locations, model="large"
    )

    # 初始化返回数据结构
    face_polys, face_masks, face_cuts = [], [], []
    img_vis = image_bgr.copy()
    H, W = img_vis.shape[:2]

    # 关键点颜色映射
    color_map = {
        "left_eye": (0, 255, 0), "right_eye": (0, 255, 0),
        "nose_bridge": (255, 0, 0), "nose_tip": (255, 0, 0),
        "top_lip": (0, 255, 255), "bottom_lip": (0, 255, 255),
        "left_eyebrow": (160, 80, 0), "right_eyebrow": (160, 80, 0),
    }

    # 处理每个检测到的人脸
    for landmark in face_landmarks_list:
        # 提取关键点
        jaw = landmark["chin"]  # 下巴轮廓点
        left_brow = landmark["left_eyebrow"]  # 左眉毛
        right_brow = landmark["right_eyebrow"]  # 右眉毛

        # 计算面部特征点
        brow_left = left_brow[0]  # 左眉起点
        brow_right = right_brow[-1]  # 右眉终点
        brow_center = np.mean(left_brow + right_brow, axis=0)  # 眉毛中心
        chin_bottom = jaw[8]  # 下巴最低点

        # 计算面部高度和偏移量
        face_height = np.linalg.norm(np.array(chin_bottom) - np.array(brow_center))
        offset = int(face_height * 0.43)  # 垂直偏移量
        offset_h_side = int(face_height * 0.30)  # 水平偏移量

        # 计算顶部曲线控制点
        vpoint_center = (int(brow_center[0]), int(brow_center[1]) - offset)
        vpoint_left = (int(brow_left[0]), int(brow_left[1]) - offset_h_side)
        vpoint_right = (int(brow_right[0]), int(brow_right[1]) - offset_h_side)

        # 生成顶部曲线点（二次贝塞尔曲线）
        top_curve_pts = []
        N = 5  # 曲线点数量
        for t in np.linspace(0, 1, N):
            # 贝塞尔曲线计算
            x = (1 - t) ** 2 * vpoint_right[0] + 2 * (1 - t) * t * vpoint_center[0] + t ** 2 * vpoint_left[0]
            y = (1 - t) ** 2 * vpoint_right[1] + 2 * (1 - t) * t * vpoint_center[1] + t ** 2 * vpoint_left[1]
            # 坐标裁剪
            x = int(np.clip(x, 0, W - 1))
            y = int(np.clip(y, 0, H - 1))
            top_curve_pts.append((x, y))

        # 组合多边形点（下巴+顶部曲线）
        poly_points = []
        poly_points.extend(jaw)  # 下巴点
        poly_points.extend(top_curve_pts)  # 顶部曲线点
        poly_points = np.array(poly_points, dtype=np.int32)
        face_polys.append(poly_points.tolist())

        # 创建人脸掩码
        mask = np.zeros(img_vis.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly_points], 255)
        face_masks.append(mask)

        # 提取人脸区域
        face_only = np.zeros_like(img_vis)
        face_only[mask > 0] = img_vis[mask > 0]
        face_cuts.append(face_only)

        # 在可视化图像上绘制人脸轮廓
        cv2.polylines(img_vis, [poly_points], True, (0, 0, 255), 2)

        # 绘制面部特征点
        for feat, pts in landmark.items():
            pts_np = np.array(pts, np.int32)
            color = color_map.get(feat, (0, 0, 255))  # 默认红色
            # 对特定特征闭合绘制
            is_closed = feat in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip']
            if pts_np.shape[0] > 1:
                cv2.polylines(img_vis, [pts_np],
                              isClosed=is_closed,
                              color=color, thickness=2)

    return face_polys, face_masks, face_cuts, img_vis