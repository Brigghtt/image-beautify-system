import sys
from PyQt5.QtWidgets import QApplication
from image_processing import ImageProcessingApp

if __name__ == '__main__':
    # 创建Qt应用
    app = QApplication(sys.argv)

    # 创建主窗口
    win = ImageProcessingApp()
    win.show()

    # 启动事件循环
    sys.exit(app.exec_())