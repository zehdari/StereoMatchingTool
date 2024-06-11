import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import os
import yaml

MAX_IMG_WIDTH = 640
MAX_IMG_HEIGHT = 360

class PointCloudWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.grid = None

    def initUI(self):
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=500)
        self.scatter = gl.GLScatterPlotItem(size=1)
        self.gl_widget.addItem(self.scatter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.gl_widget)
        self.setLayout(layout)

    def updatePointCloud(self, points, colors):
        self.scatter.setData(pos=points, color=colors, size=self.scatter.size)

    def setPointSize(self, size):
        self.scatter.setData(size=size)

    def toggleGrid(self):
        if self.grid is None:
            self.grid = gl.GLGridItem()
            self.gl_widget.addItem(self.grid)
        else:
            self.gl_widget.removeItem(self.grid)
            self.grid = None

class DepthMapWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QHBoxLayout()
        self.depth_map_label = QtWidgets.QLabel()
        self.depth_map_label.setAlignment(QtCore.Qt.AlignCenter)
        self.depth_map_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.left_image_label = QtWidgets.QLabel()
        self.left_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.left_image_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.layout.addWidget(self.depth_map_label, alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.left_image_label, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

    def updateDepthMap(self, depth_map, left_image):
        self.depth_map_label.setPixmap(QtGui.QPixmap.fromImage(self.convertToQImage(depth_map)))
        self.left_image_label.setPixmap(QtGui.QPixmap.fromImage(self.convertToQImage(left_image)))
        self.adjustSize()

    def convertToQImage(self, cv_img):
        height, width = cv_img.shape[:2]
        cv_img = cv_img.copy()
        if len(cv_img.shape) == 2:
            bytes_per_line = width
            return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * width
            return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()

    def toggleLeftImageVisibility(self, visible):
        self.left_image_label.setVisible(visible)

class StereoViewWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.image_window = ImageWindow()
        self.setCentralWidget(self.image_window)
        self.setWindowTitle('Stereo View')
        self.setGeometry(50, 50, 1200, 600)

    def updateImages(self, left_img, right_img):
        self.image_window.updateImages(left_img, right_img)
        self.adjustWindowSize()

    def adjustWindowSize(self):
        left_pixmap = self.image_window.left_label.pixmap()
        right_pixmap = self.image_window.right_label.pixmap()

        if left_pixmap and right_pixmap:
            width = left_pixmap.width() + right_pixmap.width()
            height = max(left_pixmap.height(), right_pixmap.height())
            self.setMinimumSize(width, height)
            self.resize(width, height)

class ImageWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QHBoxLayout()
        self.left_label = QtWidgets.QLabel()
        self.right_label = QtWidgets.QLabel()
        self.layout.addWidget(self.left_label)
        self.layout.addWidget(self.right_label)
        self.setLayout(self.layout)

    def updateImages(self, left_img, right_img):
        self.left_label.setPixmap(QtGui.QPixmap.fromImage(self.convertToQImage(left_img)))
        self.right_label.setPixmap(QtGui.QPixmap.fromImage(self.convertToQImage(right_img)))

    def convertToQImage(self, cv_img):
        height, width = cv_img.shape[:2]
        bytes_per_line = 3 * width
        cv_img = cv_img.copy()
        return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()

class UnifiedSettingsWindow(QtWidgets.QWidget):
    def __init__(self, depth_map_window, point_cloud_window, stereo_app):
        super().__init__()
        self.depth_map_window = depth_map_window
        self.point_cloud_window = point_cloud_window
        self.stereo_app = stereo_app
        self.image_visible = True
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Settings')
        self.layout = QtWidgets.QVBoxLayout()

        # Map Settings Section
        self.depth_map_settings_group = QtWidgets.QGroupBox("Map Settings")
        self.initDepthMapSettingsUI()
        self.layout.addWidget(self.depth_map_settings_group)

        # Image Settings Section
        self.image_settings_group = QtWidgets.QGroupBox("Image Settings")
        self.initImageSettingsUI()
        self.layout.addWidget(self.image_settings_group)

        # Point Cloud Settings Section
        self.point_cloud_settings_group = QtWidgets.QGroupBox("Point Cloud Settings")
        self.initPointCloudSettingsUI()
        self.layout.addWidget(self.point_cloud_settings_group)

        # Video Settings Section
        self.video_settings_group = QtWidgets.QGroupBox("Video Settings")
        self.initVideoSettingsUI()
        self.layout.addWidget(self.video_settings_group)

        self.setLayout(self.layout)
        self.updateButtonLabels()

    def initDepthMapSettingsUI(self):
        layout = QtWidgets.QVBoxLayout()

        self.depth_map_color_button = QtWidgets.QPushButton('Toggle Map Color')
        self.depth_map_color_button.setCheckable(True)
        self.depth_map_color_button.setChecked(True)
        self.depth_map_color_button.clicked.connect(self.stereo_app.toggleDepthMapColor)
        layout.addWidget(self.depth_map_color_button)

        self.depth_map_settings_group.setLayout(layout)

    def initImageSettingsUI(self):
        layout = QtWidgets.QVBoxLayout()
        
        self.toggle_image_button = QtWidgets.QPushButton('Show Left Image')
        self.toggle_image_button.clicked.connect(self.toggleImageVisibility)
        self.toggle_image_button.setCheckable(True)
        self.toggle_image_button.setChecked(True)
        layout.addWidget(self.toggle_image_button)

        self.display_mode_button = QtWidgets.QPushButton()
        self.display_mode_button.clicked.connect(self.stereo_app.toggleDisplayMode)
        layout.addWidget(self.display_mode_button)

        self.open_stereo_view_button = QtWidgets.QPushButton('Open Stereo View')
        self.open_stereo_view_button.clicked.connect(self.stereo_app.showStereoView)
        layout.addWidget(self.open_stereo_view_button)

        self.image_settings_group.setLayout(layout)

    def initPointCloudSettingsUI(self):
        layout = QtWidgets.QVBoxLayout()
        
        self.point_size_label = QtWidgets.QLabel(f"Point Size: {self.point_cloud_window.scatter.size}")
        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(self.point_cloud_window.scatter.size)
        self.point_size_slider.valueChanged.connect(self.updatePointSize)
        self.point_size_label.mousePressEvent = lambda event, s=self.point_size_slider, l=self.point_size_label, n="Point Size": self.openInputDialog(s, l, n)
        layout.addWidget(self.point_size_label)
        layout.addWidget(self.point_size_slider)

        self.point_scale_label = QtWidgets.QLabel(f"Point Scale: {int(self.stereo_app.point_scale * 1000)}")
        self.point_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_scale_slider.setMinimum(1)
        self.point_scale_slider.setMaximum(1000)
        self.point_scale_slider.setValue(int(self.stereo_app.point_scale * 1000))
        self.point_scale_slider.valueChanged.connect(self.updatePointScale)
        self.point_scale_label.mousePressEvent = lambda event, s=self.point_scale_slider, l=self.point_scale_label, n="Point Scale": self.openInputDialog(s, l, n)
        layout.addWidget(self.point_scale_label)
        layout.addWidget(self.point_scale_slider)

        self.toggle_centering_button = QtWidgets.QPushButton()
        self.toggle_centering_button.clicked.connect(self.toggleCentering)
        self.toggle_centering_button.setCheckable(True)
        self.toggle_centering_button.setChecked(True)
        layout.addWidget(self.toggle_centering_button)

        self.toggle_grid_button = QtWidgets.QPushButton()
        self.toggle_grid_button.clicked.connect(self.toggleGrid)
        self.toggle_grid_button.setCheckable(True)
        self.toggle_grid_button.setChecked(False)
        layout.addWidget(self.toggle_grid_button)

        self.point_cloud_settings_group.setLayout(layout)

    def initVideoSettingsUI(self):
        layout = QtWidgets.QVBoxLayout()

        self.auto_loop_button = QtWidgets.QPushButton()
        self.auto_loop_button.clicked.connect(self.toggleAutoLoop)
        self.auto_loop_button.setCheckable(True)
        self.auto_loop_button.setChecked(False)
        layout.addWidget(self.auto_loop_button)

        self.video_settings_group.setLayout(layout)

    def openInputDialog(self, slider, label, name):
        value, ok = QtWidgets.QInputDialog.getInt(self, f'Set {name}', f'Enter {name} value:', slider.value(), slider.minimum(), slider.maximum())
        if ok:
            slider.setValue(value)
            label.setText(f"{name}: {value}")

    def toggleImageVisibility(self):
        self.image_visible = not self.image_visible
        self.depth_map_window.toggleLeftImageVisibility(self.image_visible)
        self.updateButtonLabels()

    def updatePointSize(self, value):
        self.point_size_label.setText(f"Point Size: {value}")
        self.point_cloud_window.setPointSize(value)

    def updatePointScale(self, value):
        scale_factor = value * 0.001
        self.point_scale_label.setText(f"Point Scale: {value}")
        self.stereo_app.point_scale = scale_factor
        self.stereo_app.updateDisparity()

    def toggleCentering(self):
        self.stereo_app.center_points = not self.stereo_app.center_points
        self.stereo_app.updateDisparity()
        self.updateButtonLabels()

    def toggleGrid(self):
        self.point_cloud_window.toggleGrid()
        self.updateButtonLabels()

    def toggleAutoLoop(self):
        self.stereo_app.auto_loop = not self.stereo_app.auto_loop
        self.stereo_app.video_slider.updatePlayPauseButton()  # Add this line to update the button
        self.updateButtonLabels()

    def updateButtonLabels(self):
        self.toggle_image_button.setText(f'Show Left Image')
        self.display_mode_button.setText(f'Display Mode: {self.stereo_app.display_mode}')
        self.toggle_centering_button.setText(f'Center Points')
        self.toggle_grid_button.setText(f'Show Grid')
        self.auto_loop_button.setText(f'Auto Loop')

class VideoSlider(QtWidgets.QWidget):
    def __init__(self, stereo_app):
        super().__init__()
        self.stereo_app = stereo_app
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QHBoxLayout()

        self.play_pause_button = QtWidgets.QPushButton('Play')
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)

        self.play_pause_button.clicked.connect(self.togglePlayPause)
        self.slider.sliderReleased.connect(self.updateFramePosition)
        self.slider.sliderPressed.connect(self.pauseVideo)

        layout.addWidget(self.play_pause_button)
        layout.addWidget(self.slider)

        self.setLayout(layout)

    def updateFramePosition(self):   
        position = self.slider.value()
        if position >= self.slider.maximum():
            if self.stereo_app.auto_loop:
                self.stereo_app.restartVideo()
            else:
                self.stereo_app.pauseVideo()
        else:
            self.stereo_app.setFramePosition(position)
            if self.stereo_app.playing:
                self.stereo_app.playVideo()
        self.updatePlayPauseButton()


    def pauseVideo(self):
        self.stereo_app.pauseVideo()
        self.updatePlayPauseButton()

    def togglePlayPause(self):
        if self.play_pause_button.text() == 'Restart':
            self.stereo_app.restartVideo()
        else:
            if self.stereo_app.playing:
                self.stereo_app.pauseVideo()
            else:
                self.stereo_app.playVideo()
        self.updatePlayPauseButton()

    def setSliderRange(self, max_value):
        self.slider.setMaximum(max_value)

    def updateSliderPosition(self, position):
        self.slider.setValue(position)
        self.updatePlayPauseButton()

    def updatePlayPauseButton(self):
        if self.stereo_app.auto_loop:
            self.play_pause_button.setText('Pause' if self.stereo_app.playing else 'Play')
        else:
            if self.slider.value() >= self.slider.maximum():
                self.play_pause_button.setText('Restart')
            else:
                self.play_pause_button.setText('Pause' if self.stereo_app.playing else 'Play')

class StereoVisionApp(QtWidgets.QMainWindow):
    MODES = [cv2.STEREO_SGBM_MODE_SGBM, cv2.STEREO_SGBM_MODE_HH, cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_HH4]
    MODE_NAMES = ["SGBM", "HH", "SGBM_3WAY", "HH4"]
    PREFILTER_TYPES = [cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE, cv2.STEREO_BM_PREFILTER_XSOBEL]
    PREFILTER_TYPE_NAMES = ["NORMALIZED_RESPONSE", "XSOBEL"]

    def __init__(self):
        super().__init__()

        self.current_index = 0
        self.display_mode = "RGB"
        self.depth_map_color = True
        self.use_sgbm = True 
        self.use_wls = False
        self.center_points = True 
        self.point_scale = 0.05  
        self.scaling_factor = 1.0
        self.show_depth_map = True 
        self.playing = False
        self.auto_loop = False
        self.current_file = None

        self.point_cloud_window = PointCloudWindow()
        self.depth_map_window = DepthMapWindow()
        self.stereo_view_window = StereoViewWindow()
        self.unified_settings_window = UnifiedSettingsWindow(self.depth_map_window, self.point_cloud_window, self)
        self.video_slider = VideoSlider(self)

        self.loadConfig()
        if not self.use_live_feed:
            self.loadImagePaths()
        self.initUI()
        self.loadCameraParams()
        self.loadImagePair()
        self.initData()
        self.image_resized = False
        self.updateDisparity()

    def loadConfig(self):
        with open("config/config.yml", "r") as file:
            self.config = yaml.safe_load(file)
        self.left_folder = self.config.get("left_folder", "")
        self.right_folder = self.config.get("right_folder", "")
        self.baseline = self.config["baseline"]
        self.is_bgr = self.config.get("BGR", False)
        self.use_live_feed = self.config.get("Live", False)

    def initUI(self):

        # Create the menu bar
        menubar = self.menuBar()

        # Add "File" menu
        fileMenu = menubar.addMenu('File')

        # Add "Save" action
        self.saveAction = QtWidgets.QAction('Save', self)
        self.saveAction.setEnabled(False)
        self.saveAction.triggered.connect(self.saveParameters)
        fileMenu.addAction(self.saveAction)

        # Add "Save As" action
        saveAsAction = QtWidgets.QAction('Save As', self)
        saveAsAction.triggered.connect(self.saveAsParameters)
        fileMenu.addAction(saveAsAction)

        # Add "Load Parameters" action
        loadAction = QtWidgets.QAction('Load Parameters', self)
        loadAction.triggered.connect(self.loadParameters)
        fileMenu.addAction(loadAction)

         # Add "Settings" menu
        settingsMenu = menubar.addMenu('Settings')

        # Add actions to "Settings" menu
        openSettingsAction = QtWidgets.QAction('Open Settings', self)
        openSettingsAction.triggered.connect(self.showSettings)
        settingsMenu.addAction(openSettingsAction)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QtWidgets.QHBoxLayout(central_widget)

        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        self.left_panel.setMinimumWidth(300)
        self.slider_layout = QtWidgets.QVBoxLayout(self.left_panel)

        # Create sliders for disparity parameters
        self.slider_params = {
            "numDisparities": (1, 32, 8),
            "blockSize": (1, 255, 5),  # Adjusted to valid range for both SGBM and BM
            "preFilterCap": (1, 62, 5),
            "uniquenessRatio": (1, 100, 15),
            "speckleRange": (0, 100, 0),
            "speckleWindowSize": (0, 25, 3),
            "disp12MaxDiff": (0, 25, 5),
            "minDisparity": (1, 25, 1),
            "P1": (1, 3000, 216),
            "P2": (1, 12000, 864),
            "textureThreshold": (0, 100, 10),  # Only used for BM
            "preFilterSize": (5, 255, 9),
            "lambda": (8000, 80000, 8000),
            "sigma": (1, 500, 150)
        }

        self.sliders = {}
        self.slider_labels = {}
        for name, (min_val, max_val, default) in self.slider_params.items():
            slider_label = QtWidgets.QLabel(f"{name}: {default}")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            slider.valueChanged.connect(self.updateDisparity)
            slider.valueChanged.connect(lambda value, label=slider_label, name=name: label.setText(f"{name}: {value}"))
            slider_label.mousePressEvent = lambda event, s=slider, l=slider_label, n=name: self.openInputDialog(s, l, n)
            self.slider_layout.addWidget(slider_label)
            self.slider_layout.addWidget(slider)
            self.sliders[name] = slider
            self.slider_labels[name] = slider_label

        self.mode_label = QtWidgets.QLabel(f"mode: {self.MODE_NAMES[0]}")
        self.mode_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mode_slider.setMinimum(0)
        self.mode_slider.setMaximum(len(self.MODES) - 1)
        self.mode_slider.setValue(0)
        self.mode_slider.valueChanged.connect(self.updateDisparity)
        self.mode_slider.valueChanged.connect(lambda value: self.mode_label.setText(f"mode: {self.MODE_NAMES[value]}"))
        self.mode_label.mousePressEvent = lambda event, s=self.mode_slider, l=self.mode_label, n="mode": self.openInputDialog(s, l, n)
        self.slider_layout.addWidget(self.mode_label)
        self.slider_layout.addWidget(self.mode_slider)

        self.prefiltertype_label = QtWidgets.QLabel(f"preFilterType: {self.PREFILTER_TYPE_NAMES[0]}")
        self.prefiltertype_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.prefiltertype_slider.setMinimum(0)
        self.prefiltertype_slider.setMaximum(len(self.PREFILTER_TYPES) - 1)
        self.prefiltertype_slider.setValue(0)
        self.prefiltertype_slider.valueChanged.connect(self.updateDisparity)
        self.prefiltertype_slider.valueChanged.connect(lambda value: self.prefiltertype_label.setText(f"preFilterType: {self.PREFILTER_TYPE_NAMES[value]}"))
        self.prefiltertype_label.mousePressEvent = lambda event, s=self.prefiltertype_slider, l=self.prefiltertype_label, n="preFilterType": self.openInputDialog(s, l, n)
        self.slider_layout.addWidget(self.prefiltertype_label)
        self.slider_layout.addWidget(self.prefiltertype_slider)

        self.doffs_label = QtWidgets.QLabel("doffs: 0")
        self.doffs_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.doffs_slider.setMinimum(0)
        self.doffs_slider.setMaximum(1000)
        self.doffs_slider.setValue(0)
        self.doffs_slider.valueChanged.connect(self.updateDisparity)
        self.doffs_slider.valueChanged.connect(lambda value: self.doffs_label.setText(f"doffs: {value}"))
        self.doffs_label.mousePressEvent = lambda event, s=self.doffs_slider, l=self.doffs_label, n="doffs": self.openInputDialog(s, l, n)
        self.slider_layout.addWidget(self.doffs_label)
        self.slider_layout.addWidget(self.doffs_slider)

        button_layout = QtWidgets.QHBoxLayout()

        self.slider_layout.addLayout(button_layout)

        self.main_layout.addWidget(self.left_panel)

        self.right_layout = QtWidgets.QVBoxLayout()
        
        self.button_panel = QtWidgets.QWidget()
        self.button_layout = QtWidgets.QHBoxLayout(self.button_panel)

        self.prev_button = QtWidgets.QPushButton('Previous')
        self.next_button = QtWidgets.QPushButton('Next')
        self.prev_button.clicked.connect(self.prevImage)
        self.next_button.clicked.connect(self.nextImage)
        self.button_layout.addWidget(self.prev_button)
        self.button_layout.addWidget(self.next_button)

        self.toggle_map_button = QtWidgets.QPushButton()
        self.toggle_map_button.clicked.connect(self.toggleMap)
        self.button_layout.addWidget(self.toggle_map_button)

        self.toggle_algorithm_button = QtWidgets.QPushButton()
        self.toggle_algorithm_button.clicked.connect(self.toggleAlgorithm)
        self.button_layout.addWidget(self.toggle_algorithm_button)

        self.toggle_wls_button = QtWidgets.QPushButton()
        self.toggle_wls_button.clicked.connect(self.toggleWLS)
        self.toggle_wls_button.setCheckable(True)
        self.button_layout.addWidget(self.toggle_wls_button)

        self.right_layout.addWidget(self.button_panel, 0, QtCore.Qt.AlignTop)
        self.right_layout.addWidget(self.video_slider, 0, QtCore.Qt.AlignTop)
        self.right_layout.addWidget(self.depth_map_window, 0, QtCore.Qt.AlignTop)
        self.right_layout.addWidget(self.point_cloud_window, 1)

        self.main_layout.addLayout(self.right_layout, 1)

        if self.use_live_feed or len(self.image_pairs) <= 1:
            self.prev_button.hide()
            self.next_button.hide()

        self.updateSliders()
        self.updateButtonLabels()

    def openInputDialog(self, slider, label, name):
        value, ok = QtWidgets.QInputDialog.getInt(self, f'Set {name}', f'Enter {name} value:', slider.value(), slider.minimum(), slider.maximum())
        if ok:
            slider.setValue(value)
            label.setText(f"{name}: {value}")

    def showSettings(self):
        if self.unified_settings_window.isVisible():
            self.unified_settings_window.raise_()
            self.unified_settings_window.activateWindow()
        else:
            self.unified_settings_window.show()

    def showStereoView(self):
        if self.stereo_view_window.isVisible():
            self.stereo_view_window.raise_()
            self.stereo_view_window.activateWindow()
        else:
            self.stereo_view_window.show()

    def loadImagePaths(self):
        if not self.use_live_feed:
            left_files = sorted([os.path.join(self.left_folder, f) for f in os.listdir(self.left_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi'))])
            right_files = sorted([os.path.join(self.right_folder, f) for f in os.listdir(self.right_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi'))])
            self.image_pairs = list(zip(left_files, right_files))

            if len(self.image_pairs) == 0:
                print("Error: No images/videos found in the specified folders.")
                exit()

    def detect_image_format(self, image):
        if len(image.shape) == 2:
            return "Grayscale"
        elif len(image.shape) == 3 and image.shape[2] == 3:
            if self.is_bgr:
                return "BGR"
            else:
                return "RGB"
        else:
            return "Unknown"

    def loadImagePair(self):
        if self.use_live_feed:
            self.left_cap = cv2.VideoCapture(0)
            self.right_cap = cv2.VideoCapture(1)
            if not self.left_cap.isOpened() or not self.right_cap.isOpened():
                print("Error: Unable to open camera feeds")
                exit()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.updateLiveFeed)
            self.timer.start(30)  
            # Initialize a single frame for calibration purposes
            ret_left, self.left_img = self.left_cap.read()
            ret_right, self.right_img = self.right_cap.read()
            if not ret_left or not ret_right:
                print("Error: Unable to capture initial frames")
                exit()

        else:
            left_file, right_file = self.image_pairs[self.current_index]

            if left_file.endswith(('.mp4', '.avi')) and right_file.endswith(('.mp4', '.avi')):
                self.left_cap = cv2.VideoCapture(left_file)
                self.right_cap = cv2.VideoCapture(right_file)
                if not self.left_cap.isOpened() or not self.right_cap.isOpened():
                    print("Error: Unable to open video files")
                    exit()
                self.timer = QtCore.QTimer(self)
                self.timer.timeout.connect(self.updateLiveFeed)
                # Initialize a single frame for calibration purposes
                ret_left, self.left_img = self.left_cap.read()
                ret_right, self.right_img = self.right_cap.read()
                if not ret_left or not ret_right:
                    print("Error: Unable to capture initial frames")
                    exit()
                self.video_slider.setSliderRange(int(self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                self.video_slider.show()
            else:
                self.left_img = cv2.imread(left_file)
                self.right_img = cv2.imread(right_file)
                self.video_slider.hide()

                if self.left_img is None or self.right_img is None:
                    print(f"Error: One or both images not found. Please check the file paths: {left_file}, {right_file}")
                    exit()

        self.left_img = self.color_image(self.left_img)
        self.right_img = self.color_image(self.right_img)

        original_shape = self.left_img.shape

        self.left_img = self.resize_image(self.left_img, MAX_IMG_HEIGHT, MAX_IMG_WIDTH)
        self.right_img = self.resize_image(self.right_img, MAX_IMG_HEIGHT, MAX_IMG_WIDTH)

        if self.left_img.shape != original_shape:
            self.image_resized = True
        else:
            self.image_resized = False

    def color_image(self, img):
        img_format = self.detect_image_format(img)
        if img_format == "Grayscale":
            self.display_mode = "Greyscale"
            self.unified_settings_window.display_mode_button.setEnabled(False)
        else:
            self.unified_settings_window.display_mode_button.setEnabled(True)
            if img_format == "BGR" and self.is_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img_format == "RGB" and not self.is_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img

    def resize_image(self, img, max_height, max_width):
        height, width = img.shape[:2]
        if width > max_width or height > max_height:
            self.scaling_factor = min(max_width / width, max_height / height)
            img = cv2.resize(img, (int(width * self.scaling_factor), int(height * self.scaling_factor)))
        else:
            self.scaling_factor = 1.0

        return img

    def loadCameraParams(self):
        fs_left = cv2.FileStorage(self.config["left_camera_config"], cv2.FILE_STORAGE_READ)
        self.K_left = fs_left.getNode("camera_matrix").mat()
        self.dist_coeffs_left = fs_left.getNode("dist_coeffs").mat()
        self.focal_length = self.K_left[0, 0]  
        fs_left.release()

        fs_right = cv2.FileStorage(self.config["right_camera_config"], cv2.FILE_STORAGE_READ)
        self.K_right = fs_right.getNode("camera_matrix").mat()
        self.dist_coeffs_right = fs_right.getNode("dist_coeffs").mat()
        fs_right.release()

        self.R = np.eye(3, dtype=np.float64)
        self.T = np.array([[self.baseline], [0], [0]], dtype=np.float64)

    def initData(self):
        # Make sure images are loaded before initializing data
        if self.use_live_feed and (self.left_img is None or self.right_img is None):
            return

        # Get the image size
        image_size = self.left_img.shape[1::-1]  # (width, height)

        self.K_left[0, 0] *= self.scaling_factor
        self.K_left[1, 1] *= self.scaling_factor
        self.K_left[0, 2] *= self.scaling_factor
        self.K_left[1, 2] *= self.scaling_factor

        self.K_right[0, 0] *= self.scaling_factor
        self.K_right[1, 1] *= self.scaling_factor
        self.K_right[0, 2] *= self.scaling_factor
        self.K_right[1, 2] *= self.scaling_factor

        # Calculate the optimal new camera matrices
        self.K_left_optimal, self.roi_left = cv2.getOptimalNewCameraMatrix(self.K_left, self.dist_coeffs_left, image_size, 1, image_size)
        self.K_right_optimal, self.roi_right = cv2.getOptimalNewCameraMatrix(self.K_right, self.dist_coeffs_right, image_size, 1, image_size)

        # Perform stereo rectification
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K_left_optimal, self.dist_coeffs_left, 
            self.K_right_optimal, self.dist_coeffs_right, 
            image_size, self.R, self.T, 
            alpha=0
        )

        # Initialize undistort rectify maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.K_left, self.dist_coeffs_left, None, self.K_left_optimal, 
            image_size, cv2.CV_32FC1
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.K_right, self.dist_coeffs_right, None, self.K_right_optimal, 
            image_size, cv2.CV_32FC1
        )
      
    def updateLiveFeed(self):
        ret_left, left_img = self.left_cap.read()
        ret_right, right_img = self.right_cap.read()
        if not ret_left or not ret_right:
            if self.auto_loop:
                self.restartVideo()
            else:
                self.pauseVideo()
            return

        left_img = self.color_image(left_img)
        right_img = self.color_image(right_img)
        self.left_img = self.resize_image(left_img, MAX_IMG_HEIGHT, MAX_IMG_WIDTH)
        self.right_img = self.resize_image(right_img, MAX_IMG_HEIGHT, MAX_IMG_WIDTH)
        
        self.updateDisparity()
        self.updateSliderPosition()

    def updateSliderPosition(self):
        if self.left_cap:
            position = int(self.left_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.video_slider.updateSliderPosition(position)

    def toggleDisplayMode(self):
        modes = ["RGB", "BGR", "Greyscale"]
        current_index = modes.index(self.display_mode)
        self.display_mode = modes[(current_index + 1) % len(modes)]
        self.updateDisparity()
        self.updateButtonLabels()

    def toggleDepthMapColor(self):
        self.depth_map_color = not self.depth_map_color
        self.updateDisparity()
        self.updateButtonLabels()

    def toggleMap(self):
        self.show_depth_map = not self.show_depth_map
        self.updateDisparity()
        self.updateButtonLabels()

    def toggleAlgorithm(self):
        self.use_sgbm = not self.use_sgbm
        self.updateSliders()
        self.updateDisparity()
        self.updateButtonLabels()

    def toggleWLS(self):
        self.use_wls = not self.use_wls
        self.updateSliders()
        self.updateDisparity()
        self.updateButtonLabels()

    def updateSliders(self):
        if self.use_sgbm:
            self.sliders["blockSize"].setMinimum(1)
            self.sliders["blockSize"].setMaximum(50)
            self.sliders["P1"].setEnabled(True)
            self.sliders["P2"].setEnabled(True)
            self.mode_slider.setEnabled(True)
            self.sliders["textureThreshold"].setEnabled(False)
            self.sliders["preFilterSize"].setEnabled(False)
            self.prefiltertype_slider.setEnabled(False)

            self.slider_labels["P1"].setEnabled(True)
            self.slider_labels["P2"].setEnabled(True)
            self.mode_label.setEnabled(True)
            self.slider_labels["textureThreshold"].setEnabled(False)
            self.slider_labels["preFilterSize"].setEnabled(False)
            self.prefiltertype_label.setEnabled(False)

        else:
            self.sliders["blockSize"].setMinimum(5)
            self.sliders["blockSize"].setMaximum(255)
            self.sliders["P1"].setEnabled(False)
            self.sliders["P2"].setEnabled(False)
            self.mode_slider.setEnabled(False)
            self.sliders["textureThreshold"].setEnabled(True)
            self.sliders["preFilterSize"].setEnabled(True)
            self.prefiltertype_label.setEnabled(True)
            self.prefiltertype_slider.setEnabled(True)

            self.slider_labels["P1"].setEnabled(False)
            self.slider_labels["P2"].setEnabled(False)
            self.mode_label.setEnabled(False)
            self.slider_labels["textureThreshold"].setEnabled(True)
            self.slider_labels["preFilterSize"].setEnabled(True)
            self.prefiltertype_label.setEnabled(True)
    
        if self.use_wls:
            self.sliders["lambda"].setEnabled(True)
            self.sliders["sigma"].setEnabled(True)

            self.slider_labels["lambda"].setEnabled(True)
            self.slider_labels["sigma"].setEnabled(True)
        else:
            self.sliders["lambda"].setEnabled(False)
            self.sliders["sigma"].setEnabled(False)

            self.slider_labels["lambda"].setEnabled(False)
            self.slider_labels["sigma"].setEnabled(False)

    def updateDisparity(self):
        self.rectified_left = cv2.remap(self.left_img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        self.rectified_right = cv2.remap(self.right_img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        x_left, y_left, w_left, h_left = self.roi_left

        self.rectified_left = self.rectified_left[y_left:y_left+h_left, x_left:x_left+w_left]
        self.rectified_right = self.rectified_right[y_left:y_left+h_left, x_left:x_left+w_left]

        min_disparity = self.sliders["minDisparity"].value()
        num_disparities = self.sliders["numDisparities"].value() * 16
        block_size = self.sliders["blockSize"].value() | 1 
        uniqueness_ratio = self.sliders["uniquenessRatio"].value()
        speckle_window_size = self.sliders["speckleWindowSize"].value()
        speckle_range = self.sliders["speckleRange"].value()
        disp12_max_diff = self.sliders["disp12MaxDiff"].value()
        pre_filter_cap = self.sliders["preFilterCap"].value()
        P1 = self.sliders["P1"].value()
        P2 = self.sliders["P2"].value()
        texture_threshold = self.sliders["textureThreshold"].value()
        pre_filter_size = self.sliders["preFilterSize"].value() | 1
        pre_filter_type = self.PREFILTER_TYPES[self.prefiltertype_slider.value()]
        mode = self.MODES[self.mode_slider.value()]
        doffs = self.doffs_slider.value()
        lmbda = self.sliders["lambda"].value()
        sigma = self.sliders["sigma"].value() / 100

        left_img_gray = cv2.cvtColor(self.rectified_left, cv2.COLOR_BGR2GRAY)
        right_img_gray = cv2.cvtColor(self.rectified_right, cv2.COLOR_BGR2GRAY)

        if self.use_sgbm:
            stereo = cv2.StereoSGBM_create(
                minDisparity=min_disparity,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=P1,
                P2=P2,
                disp12MaxDiff=disp12_max_diff,
                uniquenessRatio=uniqueness_ratio,
                speckleWindowSize=speckle_window_size,
                speckleRange=speckle_range,
                preFilterCap=pre_filter_cap,
                mode=mode
            )
        else:
            stereo = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
            stereo.setPreFilterType(pre_filter_type)
            stereo.setPreFilterSize(pre_filter_size)
            stereo.setPreFilterCap(pre_filter_cap)
            stereo.setTextureThreshold(texture_threshold)
            stereo.setUniquenessRatio(uniqueness_ratio)
            stereo.setSpeckleWindowSize(speckle_window_size)
            stereo.setSpeckleRange(speckle_range)
            stereo.setDisp12MaxDiff(disp12_max_diff)

        right_stereo = cv2.ximgproc.createRightMatcher(stereo)
        disparity_map = stereo.compute(left_img_gray, right_img_gray).astype(np.float32) / 16.0

        if self.use_wls:
            disparity_map_right = right_stereo.compute(right_img_gray, left_img_gray).astype(np.float32) / 16.0
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
            disparity_map = wls_filter.filter(disparity_map, left_img_gray, disparity_map_right=disparity_map_right)
        else:
            disparity_map_right = np.zeros(disparity_map.shape, dtype=np.float32)

        disparity_map[disparity_map < min_disparity] = min_disparity 
        depth_map = np.zeros(disparity_map.shape, dtype=np.float32)
        epsilon = 1e-6  # Small value to prevent division by zero
        valid_disparity_mask = disparity_map > (min_disparity + epsilon)
        depth_map[valid_disparity_mask] = self.baseline * self.focal_length / (disparity_map[valid_disparity_mask] + doffs + epsilon)

        # Display map
        if self.show_depth_map:
            display_map = depth_map
        else:
            display_map = disparity_map

        display_map = np.nan_to_num(display_map, nan=0.0, posinf=255.0, neginf=0.0)  # Handle invalid values

        if self.depth_map_color:
            display_map_normalized = cv2.normalize(display_map, None, 0, 255, cv2.NORM_MINMAX)
            display_map_normalized = np.uint8(display_map_normalized)
            display_map_colored = cv2.applyColorMap(display_map_normalized, cv2.COLORMAP_JET)
            self.depth_map_window.updateDepthMap(display_map_colored, self.getImageForDisplayMode(self.rectified_left))
        else:
            display_map_normalized = cv2.normalize(display_map, None, 0, 255, cv2.NORM_MINMAX)
            display_map_normalized = np.uint8(display_map_normalized)
            self.depth_map_window.updateDepthMap(display_map_normalized, self.getImageForDisplayMode(self.rectified_left))

        points = cv2.reprojectImageTo3D(depth_map, self.Q)
        
        colors_bgr = self.rectified_left

        mask = disparity_map > min_disparity
        out_points = points[mask]

        if out_points.size == 0:
            return

        if self.display_mode == "RGB":
            out_colors = colors_bgr[mask]
        elif self.display_mode == "BGR":
            out_colors = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)[mask]
        elif self.display_mode == "Greyscale":
            colors_gray = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2GRAY)
            out_colors = cv2.cvtColor(colors_gray, cv2.COLOR_GRAY2RGB)[mask]

        # Rotate points if needed
        rotation_matrix_x = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]], dtype=np.float64)
        out_points = out_points.dot(rotation_matrix_x.T)  

        # Center the mean of the points
        if self.center_points and out_points.size > 0:
            mean = np.mean(out_points, axis=0)
            out_points -= mean

        scaling_factor = self.point_scale
        out_points = out_points[::10] * scaling_factor
        out_colors = out_colors[::10] / 255.0

        self.point_cloud_window.updatePointCloud(out_points, out_colors)

        # Update rectified images
        self.stereo_view_window.updateImages(self.getImageForDisplayMode(self.rectified_left), self.getImageForDisplayMode(self.rectified_right))

    def getImageForDisplayMode(self, image):
        if self.display_mode == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.display_mode == "Greyscale":
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        else:
            return image

    def nextImage(self):
        if self.playing:
            self.pauseVideo()
        self.video_slider.slider.setValue(0)
        if self.current_index < len(self.image_pairs) - 1:
            self.current_index += 1
            self.loadImagePair()
            self.updateDisparity()
        self.updateButtonLabels()

    def prevImage(self):
        if self.playing:
            self.pauseVideo()
        self.video_slider.slider.setValue(0)
        if self.current_index > 0:
            self.current_index -= 1
            self.loadImagePair()
            self.updateDisparity()
        self.updateButtonLabels()

    def playVideo(self):
        self.playing = True
        self.timer.start(30)  # Assuming 30 FPS
        if self.left_cap and self.right_cap:
            current_frame = self.video_slider.slider.value()
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    def pauseVideo(self):
        self.playing = False
        self.timer.stop()
    
    def restartVideo(self):
        if self.left_cap and self.right_cap:
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.updateSliderPosition()
            self.playVideo()

    def setFramePosition(self, position):
        if self.left_cap and self.right_cap:
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret_left, self.left_img = self.left_cap.read()
            ret_right, self.right_img = self.right_cap.read()
            if ret_left and ret_right:
                self.left_img = self.color_image(self.left_img) 
                self.right_img = self.color_image(self.right_img)
                self.left_img = self.resize_image(self.left_img, MAX_IMG_HEIGHT, MAX_IMG_WIDTH)
                self.right_img = self.resize_image(self.right_img, MAX_IMG_HEIGHT, MAX_IMG_WIDTH)
                self.updateDisparity()

    def updateButtonLabels(self):
        self.toggle_algorithm_button.setText(f'Algorithm: {"SGBM" if self.use_sgbm else "BM"}')
        self.toggle_wls_button.setText(f'WLS Filter')
        self.toggle_map_button.setText(f'Showing: {"Depth Map" if self.show_depth_map else "Disparity Map"}')
        if not self.use_live_feed:
            self.prev_button.setEnabled(self.current_index > 0)
            self.next_button.setEnabled(self.current_index < len(self.image_pairs) - 1)
        self.unified_settings_window.updateButtonLabels()
        self.video_slider.updatePlayPauseButton()

    def loadParameters(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Parameters", "", "YAML Files (*.yml);;All Files (*)", options=options)
        if file_name:
            self.loadParametersFromFile(file_name)

    def applyParameters(self, parameters):
        self.sliders["numDisparities"].setValue(parameters.get("numDisparities", self.sliders["numDisparities"].value()))
        self.sliders["blockSize"].setValue(parameters.get("blockSize", self.sliders["blockSize"].value()))
        self.sliders["preFilterCap"].setValue(parameters.get("preFilterCap", self.sliders["preFilterCap"].value()))
        self.sliders["uniquenessRatio"].setValue(parameters.get("uniquenessRatio", self.sliders["uniquenessRatio"].value()))
        self.sliders["speckleRange"].setValue(parameters.get("speckleRange", self.sliders["speckleRange"].value()))
        self.sliders["speckleWindowSize"].setValue(parameters.get("speckleWindowSize", self.sliders["speckleWindowSize"].value()))
        self.sliders["disp12MaxDiff"].setValue(parameters.get("disp12MaxDiff", self.sliders["disp12MaxDiff"].value()))
        self.sliders["minDisparity"].setValue(parameters.get("minDisparity", self.sliders["minDisparity"].value()))
        self.sliders["P1"].setValue(parameters.get("P1", self.sliders["P1"].value()))
        self.sliders["P2"].setValue(parameters.get("P2", self.sliders["P2"].value()))
        self.sliders["textureThreshold"].setValue(parameters.get("textureThreshold", self.sliders["textureThreshold"].value()))
        self.sliders["preFilterSize"].setValue(parameters.get("preFilterSize", self.sliders["preFilterSize"].value()))
        self.sliders["lambda"].setValue(parameters.get("lambda", self.sliders["lambda"].value()))
        self.sliders["sigma"].setValue(parameters.get("sigma", self.sliders["sigma"].value()))
        self.mode_slider.setValue(self.MODE_NAMES.index(parameters.get("mode", self.MODE_NAMES[self.mode_slider.value()])))
        self.prefiltertype_slider.setValue(self.PREFILTER_TYPE_NAMES.index(parameters.get("preFilterType", self.PREFILTER_TYPE_NAMES[self.prefiltertype_slider.value()])))
        self.doffs_slider.setValue(parameters.get("doffs", self.doffs_slider.value()))
        self.depth_map_color = parameters.get("depthMapColor", self.depth_map_color)
        self.use_sgbm = parameters.get("useSGBM", self.use_sgbm)
        self.use_wls = parameters.get("useWLS", self.use_wls)
        self.center_points = parameters.get("centerPoints", self.center_points)
        self.point_scale = parameters.get("pointScale", self.point_scale)
        self.display_mode = parameters.get("displayMode", self.display_mode)
        self.show_depth_map = parameters.get("showDepthMap", self.show_depth_map)

        self.updateSliders()
        self.updateDisparity()
        self.updateButtonLabels()

    def saveParameters(self):
        if self.current_file:
            settings = self.collectParameters()
            self.saveParametersToFile(settings, self.current_file)
        else:
            self.saveAsParameters()

    def saveAsParameters(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Parameters", "", "YAML Files (*.yml);;All Files (*)", options=options)
        if file_path:
            if not file_path.endswith(".yml"):
                file_path += ".yml"
            settings = self.collectParameters()
            self.saveParametersToFile(settings, file_path)
            self.saveAction.setEnabled(True)

    def loadParametersFromFile(self, file_path):
        with open(file_path, 'r') as file:
            parameters = yaml.safe_load(file)
            self.applyParameters(parameters)
            self.saveAction.setEnabled(True)

    def collectParameters(self):
        parameters = {
            "numDisparities": self.sliders["numDisparities"].value(),
            "blockSize": self.sliders["blockSize"].value(),
            "preFilterCap": self.sliders["preFilterCap"].value(),
            "uniquenessRatio": self.sliders["uniquenessRatio"].value(),
            "speckleRange": self.sliders["speckleRange"].value(),
            "speckleWindowSize": self.sliders["speckleWindowSize"].value(),
            "disp12MaxDiff": self.sliders["disp12MaxDiff"].value(),
            "minDisparity": self.sliders["minDisparity"].value(),
            "P1": self.sliders["P1"].value(),
            "P2": self.sliders["P2"].value(),
            "textureThreshold": self.sliders["textureThreshold"].value(),
            "preFilterSize": self.sliders["preFilterSize"].value(),
            "lambda": self.sliders["lambda"].value(),
            "sigma": self.sliders["sigma"].value(),
            "mode": self.MODE_NAMES[self.mode_slider.value()],
            "preFilterType": self.PREFILTER_TYPE_NAMES[self.prefiltertype_slider.value()],
            "doffs": self.doffs_slider.value(),
            "depthMapColor": self.depth_map_color,
            "useSGBM": self.use_sgbm,
            "useWLS": self.use_wls,
            "centerPoints": self.center_points,
            "pointScale": self.point_scale,
            "displayMode": self.display_mode,
            "showDepthMap": self.show_depth_map,
        }
        return parameters

    def saveParametersToFile(self, parameters, file_path):
        with open(file_path, "w") as file:
            yaml.dump(parameters, file)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = StereoVisionApp()
    mainWin.setGeometry(50, 50, 1200, 900) 
    mainWin.setWindowTitle('Stereo Vision Disparity Adjustment')
    mainWin.show()
    sys.exit(app.exec_())