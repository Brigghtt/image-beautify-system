import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QTabWidget, QSplitter, QMessageBox,
    QSlider, QComboBox, QGroupBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

# 导入自定义模块
from image_utils import cartoonize_face, whiten_skin, retro_face
from face_tools import detect_faces_with_landmark, is_valid_face_roi
from gfpgan_utils import beautify_face


class ImageProcessingApp(QMainWindow):
    """主应用程序窗口，实现图像处理功能界面"""


    def __init__(self):
        super().__init__()
        self.initUI()  # 初始化UI
        # 图像相关状态变量
        self.current_image = None  # 当前显示的图像
        self.original_image = None  # 原始加载的图像
        self.image_history = []  # 图像历史记录
        # 人脸处理相关变量
        self.face_polys = []  # 人脸多边形点
        self.face_masks = []  # 人脸掩码
        self.face_cuts = []  # 裁剪出的人脸图像
        # 摄像头相关变量
        self.camera_timer = None  # 摄像头定时器
        self.camera_cap = None  # 摄像头捕获对象

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('数字图像处理系统')
        self.resize(1200, 800)

        # 设置全局样式
        self.setStyleSheet('''
        QPushButton {
            background-color: #FFA07A;     /* 浅橙色 */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 15px;
            margin: 5px;
        }
        QPushButton:hover {
            background-color: #FF8C00;    /* 深橙色 */
        }
        ''')

        # 主窗口布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # 工具栏
        tool_bar = QHBoxLayout()
        undo_btn = QPushButton("撤销")
        undo_btn.clicked.connect(self.undo_action)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_image)
        tool_bar.addWidget(undo_btn)
        tool_bar.addWidget(reset_btn)
        tool_bar.addStretch()
        main_layout.addLayout(tool_bar)

        # 标签页组件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 创建各功能标签页
        self.create_load_tab()  # 图像加载页
        self.create_preprocessing_tab()  # 预处理页
        self.create_segmentation_tab()  # 图像分割页
        self.create_face_detection_tab()  # 人脸检测页

    def init_image_label(self, label):
        """
        初始化图像显示标签

        参数:
            label (QLabel): 需要初始化的标签

        返回:
            QLabel: 初始化后的标签
        """
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background:white; min-height:400px;")
        label.setFixedSize(800, 600)  # 设置固定大小
        return label

    def create_load_tab(self):
        """创建图像加载标签页"""
        load_tab = QWidget()
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 添加功能按钮
        load_btn = QPushButton("从文件加载")
        load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(load_btn)

        self.camera_btn = QPushButton("打开摄像头")
        self.camera_btn.clicked.connect(self.open_camera)
        left_layout.addWidget(self.camera_btn)

        self.capture_btn = QPushButton("采集摄像头画面")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_from_camera)
        left_layout.addWidget(self.capture_btn)

        self.close_cam_btn = QPushButton("关闭摄像头")
        self.close_cam_btn.setEnabled(False)
        self.close_cam_btn.clicked.connect(self.close_camera)
        left_layout.addWidget(self.close_cam_btn)

        save_btn = QPushButton("保存图像")
        save_btn.clicked.connect(self.save_image)
        left_layout.addWidget(save_btn)

        left_layout.addStretch()

        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.load_label = self.init_image_label(QLabel("加载图像"))
        right_layout.addWidget(self.load_label)
        right_layout.setAlignment(Qt.AlignCenter)

        # 组装分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([256, 800])
        layout.addWidget(splitter)
        load_tab.setLayout(layout)
        self.tab_widget.addTab(load_tab, "图像加载")

    def load_image(self):
        """从文件加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", "Images (*.bmp *.jpg *.jpeg *.png)"
        )
        if file_path:
            # 读取图像文件
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            if img is None:
                QMessageBox.warning(self, "读取失败", "无法读取该图像")
                return

                # 处理不同格式的图像
                if len(img.shape) == 2:  # 灰度图转BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA转BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 更新状态
            self.current_image = img
            self.original_image = img.copy()
            self.image_history = [img.copy()]
            self.face_polys, self.face_masks, self.face_cuts = [], [], []
            self.display_image(self.current_image)

    def save_image(self):
        """保存当前图像到文件"""
        if self.current_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "Images (*.bmp *.jpg *.png)"
            )
            if file_path:
                # 根据扩展名编码图像
                ext = file_path.split(".")[-1].lower()
                result, im_buf = cv2.imencode(f'.{ext}', self.current_image)
                if result:
                    im_buf.tofile(file_path)
                else:
                    QMessageBox.warning(self, "保存失败", "保存失败，请检查路径或格式")
        else:
            QMessageBox.warning(self, "未加载", "请先加载图像")

    def open_camera(self):
        """打开摄像头进行预览"""
        if self.camera_cap is not None:
            return

        # 初始化摄像头
        self.camera_cap = cv2.VideoCapture(0)
        if not self.camera_cap.isOpened():
            QMessageBox.warning(self, "摄像头错误", "无法打开摄像头！")
            self.camera_cap = None
            return

        # 更新UI状态
        self.camera_last_frame = None
        self.camera_btn.setEnabled(False)
        self.capture_btn.setEnabled(True)
        self.close_cam_btn.setEnabled(True)

        # 设置定时器刷新预览
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_preview)
        self.camera_timer.start(30)  # ~30FPS

    def update_camera_preview(self):
        """更新摄像头预览画面"""
        if self.camera_cap is None:
            return

        # 读取摄像头帧
        ret, frame = self.camera_cap.read()
        if ret:
            self.camera_last_frame = frame  # 存储最后一帧

            # 转换并显示图像
            show_img = frame.copy()
            h, w = show_img.shape[:2]
            qimg = QImage(show_img.data, w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qimg)
            self.load_label.setPixmap(pixmap.scaled(
                self.load_label.width(), self.load_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def capture_from_camera(self):
        """从摄像头采集当前帧"""
        if self.camera_cap is None or self.camera_last_frame is None:
            return

        # 使用最后捕获的帧
        img = self.camera_last_frame.copy()
        self.current_image = img
        self.original_image = img.copy()
        self.image_history = [img.copy()]
        self.face_polys, self.face_masks, self.face_cuts = [], [], []
        self.close_camera()  # 采集后关闭摄像头
        self.display_image(self.current_image)

    def close_camera(self):
        """关闭摄像头并释放资源"""
        if self.camera_timer:
            self.camera_timer.stop()
            self.camera_timer = None
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None

        # 更新UI状态
        self.camera_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)
        self.close_cam_btn.setEnabled(False)
        self.load_label.clear()
        self.camera_last_frame = None

    def create_preprocessing_tab(self):
        """创建图像预处理标签页"""
        preprocessing_tab = QWidget()
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # --- 基础操作组 ---
        grp_base = QGroupBox("基础操作")
        base_layout = QVBoxLayout()
        grayscale_btn = QPushButton("灰度转换")
        grayscale_btn.clicked.connect(self.convert_to_grayscale)
        blur_btn = QPushButton("高斯模糊")
        blur_btn.clicked.connect(self.apply_gaussian_blur)
        base_layout.addWidget(grayscale_btn)
        base_layout.addWidget(blur_btn)
        grp_base.setLayout(base_layout)

        # --- 形态学操作组 ---
        grp_morph = QGroupBox("形态学处理")
        morph_layout = QVBoxLayout()
        morph_dilate_btn = QPushButton("形态学膨胀")
        morph_dilate_btn.clicked.connect(self.apply_morph_dilate)
        morph_erode_btn = QPushButton("形态学腐蚀")
        morph_erode_btn.clicked.connect(self.apply_morph_erode)
        morph_layout.addWidget(morph_dilate_btn)
        morph_layout.addWidget(morph_erode_btn)
        grp_morph.setLayout(morph_layout)

        # --- 特征测量组 ---
        grp_feat = QGroupBox("对象特征提取")
        feat_layout = QVBoxLayout()
        area_btn = QPushButton("测量面积")
        area_btn.clicked.connect(self.measure_area)
        centroid_btn = QPushButton("测量质心")
        centroid_btn.clicked.connect(self.measure_centroid)
        aspect_btn = QPushButton("测量长宽比")
        aspect_btn.clicked.connect(self.measure_aspect_ratio)
        feat_layout.addWidget(area_btn)
        feat_layout.addWidget(centroid_btn)
        feat_layout.addWidget(aspect_btn)
        grp_feat.setLayout(feat_layout)

        # 分隔线
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)

        # 组装左侧面板
        left_layout.addWidget(grp_base)
        left_layout.addWidget(sep1)
        left_layout.addWidget(grp_morph)
        left_layout.addWidget(sep2)
        left_layout.addWidget(grp_feat)
        left_layout.addStretch()

        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.preprocess_label = QLabel("图像预处理")
        self.preprocess_label.setAlignment(Qt.AlignCenter)
        self.preprocess_label.setStyleSheet('background:white;min-height:400px;')
        right_layout.addWidget(self.preprocess_label)

        # 组装分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 800])
        layout.addWidget(splitter)
        preprocessing_tab.setLayout(layout)
        self.tab_widget.addTab(preprocessing_tab, "图像预处理")

    def create_segmentation_tab(self):
        """创建图像分割标签页"""
        segmentation_tab = QWidget()
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # --- 区域分割组 ---
        grp_region = QGroupBox("区域分割")
        reg_layout = QVBoxLayout()
        watershed_btn = QPushButton("分水岭法")
        watershed_btn.clicked.connect(self.region_watershed)
        reg_layout.addWidget(watershed_btn)
        grp_region.setLayout(reg_layout)

        # --- 边缘检测组 ---
        grp_edge = QGroupBox("边缘检测")
        edge_layout = QVBoxLayout()
        sobel_btn = QPushButton("Sobel算子")
        sobel_btn.clicked.connect(self.edge_sobel)
        edge_btn = QPushButton("Canny算子")
        edge_btn.clicked.connect(self.edge_detection)
        edge_layout.addWidget(sobel_btn)
        edge_layout.addWidget(edge_btn)
        grp_edge.setLayout(edge_layout)

        # --- 阈值分割组 ---
        grp_thresh = QGroupBox("阈值分割")
        th_layout = QVBoxLayout()
        threshold_segmentation_btn = QPushButton("固定阈值分割")
        threshold_segmentation_btn.clicked.connect(self.threshold_segmentation)
        adaptive_btn = QPushButton("自适应阈值")
        adaptive_btn.clicked.connect(self.threshold_adaptive)
        th_layout.addWidget(threshold_segmentation_btn)
        th_layout.addWidget(adaptive_btn)
        grp_thresh.setLayout(th_layout)

        # 组装左侧面板
        left_layout.addWidget(grp_region)
        left_layout.addWidget(grp_edge)
        left_layout.addWidget(grp_thresh)
        left_layout.addStretch()

        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.segmentation_label = QLabel("图像分割")
        self.segmentation_label.setAlignment(Qt.AlignCenter)
        self.segmentation_label.setStyleSheet('background:white;min-height:400px;')
        right_layout.addWidget(self.segmentation_label)

        # 组装分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 800])
        layout.addWidget(splitter)
        segmentation_tab.setLayout(layout)
        self.tab_widget.addTab(segmentation_tab, "图像分割")

    def create_face_detection_tab(self):
        """创建人脸检测与编辑标签页"""
        face_detection_tab = QWidget()
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 人脸检测功能
        detect_face_btn = QPushButton("检测/抠出人脸")
        detect_face_btn.clicked.connect(self.detect_faces)
        left_layout.addWidget(detect_face_btn)

        # 风格化功能
        stylize_label = QLabel("人脸风格化：")
        self.edit_combo = QComboBox()
        self.edit_combo.addItems(["卡通化", "复古效果"])
        stylize_btn = QPushButton("应用风格化")
        stylize_btn.clicked.connect(self.apply_stylize)
        left_layout.addWidget(stylize_label)
        left_layout.addWidget(self.edit_combo)
        left_layout.addWidget(stylize_btn)

        # 美颜功能
        beauty_label = QLabel("模糊美颜：")
        beauty_apply_btn = QPushButton("应用美颜")
        beauty_apply_btn.clicked.connect(self.apply_beauty)
        left_layout.addWidget(beauty_label)
        left_layout.addWidget(beauty_apply_btn)

        # 美白功能
        whiten_label = QLabel("肤色美白：")
        self.whiten_slider = QSlider(Qt.Horizontal)
        self.whiten_slider.setRange(0, 50)
        self.whiten_slider.setValue(18)
        whiten_slider_label = QLabel("美白程度")
        self.whiten_slider_val = QLabel(str(self.whiten_slider.value()))
        self.whiten_slider.valueChanged.connect(lambda val: self.whiten_slider_val.setText(str(val)))
        whiten_apply_btn = QPushButton("应用美白")
        whiten_apply_btn.clicked.connect(self.apply_whiten)
        left_layout.addWidget(whiten_label)
        subw = QHBoxLayout()
        subw.addWidget(whiten_slider_label)
        subw.addWidget(self.whiten_slider)
        subw.addWidget(self.whiten_slider_val)
        left_layout.addLayout(subw)
        left_layout.addWidget(whiten_apply_btn)

        # 历史操作
        undo_btn2 = QPushButton("撤销(仅人脸编辑历史)")
        undo_btn2.clicked.connect(self.undo_action)
        left_layout.addWidget(undo_btn2)

        # 保存功能
        save_face_btn = QPushButton("保存抠出人脸")
        save_face_btn.clicked.connect(self.save_faces)
        left_layout.addWidget(save_face_btn)
        left_layout.addStretch()

        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.face_label = QLabel("人脸检测与编辑")
        self.face_label.setAlignment(Qt.AlignCenter)
        self.face_label.setStyleSheet('background:white;min-height:400px;')
        right_layout.addWidget(self.face_label)

        # 组装分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 800])
        layout.addWidget(splitter)
        face_detection_tab.setLayout(layout)
        self.tab_widget.addTab(face_detection_tab, "人脸检测")

    def display_image(self, image):
        """
        在所有标签页显示当前图像

        参数:
            image (np.ndarray): 要显示的图像
        """
        if image is None:
            return

        # 转换OpenCV图像为Qt格式
        h, w = image.shape[:2]
        qimg = QImage(image.data, w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)

        # 在各个标签页显示
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.load_label.setPixmap(scaled_pixmap)
        if hasattr(self, "preprocess_label"):
            self.preprocess_label.setPixmap(scaled_pixmap)
        if hasattr(self, "segmentation_label"):
            self.segmentation_label.setPixmap(scaled_pixmap)
        if hasattr(self, "face_label"):
            self.face_label.setPixmap(scaled_pixmap)

    def display_face(self, image):
        """显示人脸处理结果（统一调用display_image）"""
        self.display_image(image)

    def undo_action(self):
        """撤销上一步操作"""
        if len(self.image_history) > 1:
            self.image_history.pop()  # 移除当前状态
            self.current_image = self.image_history[-1].copy()  # 恢复前一个状态
            self.display_face(self.current_image)

    def reset_image(self):
        """重置图像到初始状态"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.image_history = [self.current_image.copy()]
            self.face_polys, self.face_masks, self.face_cuts = [], [], []
            self.display_face(self.current_image)

    # ===== 图像预处理函数 =====
    def convert_to_grayscale(self):
        """转换为灰度图像"""
        if self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # 保持三通道
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    def apply_gaussian_blur(self):
        """应用高斯模糊"""
        if self.current_image is not None:
            blurred_image = cv2.GaussianBlur(self.current_image, (15, 15), 0)
            self.current_image = blurred_image
            self.image_history.append(blurred_image.copy())
            self.display_image(self.current_image)

    def apply_morph_dilate(self):
        """应用形态学膨胀"""
        if self.current_image is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.dilate(self.current_image, kernel, iterations=1)
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    def apply_morph_erode(self):
        """应用形态学腐蚀"""
        if self.current_image is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.erode(self.current_image, kernel, iterations=1)
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    def measure_area(self, noise_threshold=20, min_target_area=50):
        """测量图像中目标的面积"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = self.current_image.copy()
            areas = []
            target_count = 0

            # 遍历所有轮廓
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= max(noise_threshold, min_target_area):  # 过滤掉小面积噪声和小目标
                    areas.append(area)
                    target_count += 1
                    cv2.drawContours(img, [cnt], -1, (0, 255, 255), 2)  # 绘制轮廓

            # 显示结果
            if target_count > 0:
                total_area = np.sum(areas)
                avg_area = np.mean(areas)
                QMessageBox.information(self, "面积测量",
                                        f"总面积: {int(total_area)} 像素\n目标数量：{target_count} 个\n平均面积：{int(avg_area)} 像素\n各目标面积: {[int(a) for a in areas]}")
            else:
                QMessageBox.information(self, "面积测量", "未检测到有效目标")

            # 将处理后的图像添加到历史记录并更新当前图像
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(img)

    def measure_centroid(self, min_target_area=50):
        """测量图像中目标的质心"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = self.current_image.copy()

            # 计算并绘制质心
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area >= min_target_area:  # 忽略小目标
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)  # 红色圆点标记质心
                        cv2.putText(img, str(i + 1), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),
                                    2)  # 添加目标编号

            # 将处理后的图像添加到历史记录并更新当前图像
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(img)

    def measure_aspect_ratio(self, min_ratio=0.1, max_ratio=10.0, min_target_area=50):
        """测量图像中目标的长宽比"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = self.current_image.copy()
            ratios = []
            target_count = 0

            # 计算每个目标的长宽比
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area >= min_target_area:  # 忽略小目标
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h == 0 or w == 0:  # 避免除零错误
                        continue
                    ratio = w / h
                    if min_ratio <= ratio <= max_ratio:
                        ratios.append(ratio)
                        target_count += 1
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 绘制边界框
                        cv2.putText(img, f"{ratio:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),
                                    2)  # 添加长宽比值

            # 显示结果
            if ratios:
                avg_ratio = np.mean(ratios)
                min_ratio = min(ratios)
                max_ratio = max(ratios)
                QMessageBox.information(self, "长宽比",
                                        f"平均目标长宽比: {avg_ratio:.2f}\n最小长宽比: {min_ratio:.2f}\n最大长宽比: {max_ratio:.2f}\n目标数量：{target_count} 个")
            else:
                QMessageBox.information(self, "长宽比", "未检测到有效目标")

            # 将处理后的图像添加到历史记录并更新当前图像
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(img)

    # ===== 图像分割函数 =====
    def edge_detection(self):
        """Canny边缘检测"""
        if self.current_image is not None:
            edges = cv2.Canny(self.current_image, 100, 200)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # 转为三通道显示
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    def edge_sobel(self):
        """Sobel边缘检测"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            # 计算x和y方向的梯度
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
            # 转换为8位图像
            absx = cv2.convertScaleAbs(grad_x)
            absy = cv2.convertScaleAbs(grad_y)
            # 合并梯度
            edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
            img = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # 转为三通道
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    def region_watershed(self):
        """分水岭图像分割"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            # OTSU阈值分割
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 形态学开运算去噪
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            # 确定背景区域
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            # 确定前景区域
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            # 确定未知区域
            unknown = cv2.subtract(sure_bg, sure_fg)
            # 标记连通区域
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            # 应用分水岭算法
            img = self.current_image.copy()
            cv2.watershed(img, markers)
            # 标记边界
            img[markers == -1] = [255, 0, 0]  # 蓝色边界
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    def threshold_segmentation(self):
        """固定阈值分割"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # 转为三通道
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    def threshold_adaptive(self):
        """自适应阈值分割"""
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            th = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)  # 转为三通道
            self.current_image = img
            self.image_history.append(img.copy())
            self.display_image(self.current_image)

    # ===== 人脸处理函数 =====
    def detect_faces(self):
        """检测图像中的人脸并提取特征"""
        if self.current_image is None:
            QMessageBox.warning(self, "错误", "请先加载图片")
            return

        # 调用人脸检测函数
        res = detect_faces_with_landmark(self.current_image)
        self.face_polys, self.face_masks, self.face_cuts, img_vis = res

        # 更新显示和状态
        self.display_face(img_vis)
        self.image_history.append(img_vis.copy())
        QMessageBox.information(self, "人脸检测", f"检测到{len(self.face_cuts)}张人脸")

    def apply_stylize(self):
        """应用人脸风格化效果"""
        if self.current_image is None or not self.face_masks:
            QMessageBox.warning(self, "错误", "请先检测人脸")
            return

        # 获取选择的风格
        op = self.edit_combo.currentText()
        new_img = self.current_image.copy()

        # 处理每个检测到的人脸
        for idx, msk in enumerate(self.face_masks):
            if msk is None or np.count_nonzero(msk) < 10:  # 跳过无效掩码
                continue

            # 提取人脸区域
            face_roi = cv2.bitwise_and(new_img, new_img, mask=msk)

            # 应用选择的风格
            if op == "卡通化":
                face_new = cartoonize_face(face_roi)
            elif op == "复古效果":
                face_new = retro_face(face_roi)
            else:
                face_new = face_roi

            # 替换原图中的人脸区域
            new_img[msk > 0] = face_new[msk > 0]

        # 更新显示和状态
        self.display_face(new_img)
        self.image_history.append(new_img.copy())

    def apply_beauty(self):
        """应用GFPGAN美颜效果"""
        if self.current_image is None or not self.face_masks:
            QMessageBox.warning(self, "错误", "请先检测人脸")
            return

        new_img = self.current_image.copy()

        # 处理每个检测到的人脸
        for idx, msk in enumerate(self.face_masks):
            if msk is None or np.count_nonzero(msk) < 10:  # 跳过无效掩码
                continue

            # 提取人脸区域
            face_roi = cv2.bitwise_and(new_img, new_img, mask=msk)

            # 检查ROI有效性
            if not is_valid_face_roi(face_roi):
                print(f"第{idx}个人脸ROI无效，跳过GFPGAN")
                continue

            try:
                # 应用GFPGAN美颜
                face_new = beautify_face(face_roi)
                new_img[msk > 0] = face_new[msk > 0]
            except Exception as e:
                print("GFPGAN异常: ", e)
                continue

        # 更新显示和状态
        self.display_face(new_img)
        self.image_history.append(new_img.copy())

    def apply_whiten(self):
        """应用肤色美白效果"""
        if self.current_image is None or not self.face_masks:
            QMessageBox.warning(self, "错误", "请先检测人脸")
            return

        # 获取美白强度
        yval = self.whiten_slider.value()
        new_img = self.current_image.copy()

        # 处理每个检测到的人脸
        for idx, msk in enumerate(self.face_masks):
            if msk is None or np.count_nonzero(msk) < 10:  # 跳过无效掩码
                continue

            # 提取人脸区域并美白
            face_roi = cv2.bitwise_and(new_img, new_img, mask=msk)
            face_new = whiten_skin(face_roi, y_value=yval)
            new_img[msk > 0] = face_new[msk > 0]

        # 更新显示和状态
        self.display_face(new_img)
        self.image_history.append(new_img.copy())

    def save_faces(self):
        """保存抠出的人脸图像"""
        if not self.face_cuts or not self.face_masks:
            QMessageBox.warning(self, "未检测到人脸", "请先检测/抠出人脸")
            return

        # 保存每个检测到的人脸
        for idx, (face, mask) in enumerate(zip(self.face_cuts, self.face_masks)):
            # 准备掩码
            if mask.ndim == 3:  # 确保单通道
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = (mask > 127).astype(np.uint8) * 255  # 二值化

            if face.shape[2] == 3:
                face_bgra = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)
            else:
                face_bgra = face.copy()
            face_bgra[:, :, 3] = mask  # 设置alpha通道

            # 保存为PNG
            save_path, _ = QFileDialog.getSaveFileName(
                self, f"保存人脸_{idx + 1}", f"face_{idx + 1}.png", "PNG Files (*.png)"
            )
            if save_path:
                cv2.imwrite(save_path, face_bgra)

        QMessageBox.information(self, "保存成功", "人脸已保存为PNG文件")

    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_Escape:  # ESC键退出
            self.close()