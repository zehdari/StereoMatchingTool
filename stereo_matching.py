import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import os
import yaml

class PointCloudWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=500)
        self.grid = gl.GLGridItem()
        self.gl_widget.addItem(self.grid)
        self.scatter = gl.GLScatterPlotItem(size=1)
        self.gl_widget.addItem(self.scatter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.gl_widget)
        self.setLayout(layout)

    def updatePointCloud(self, points, colors):
        self.scatter.setData(pos=points, color=colors, size=1)

class DepthMapWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.depth_map_label = QtWidgets.QLabel()
        self.depth_map_label.setAlignment(QtCore.Qt.AlignCenter)
        self.depth_map_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.layout.addWidget(self.depth_map_label, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(self.layout)

    def updateDepthMap(self, depth_map):
        self.depth_map_label.setPixmap(QtGui.QPixmap.fromImage(self.convertToQImage(depth_map)))
        self.adjustSize()

    def convertToQImage(self, cv_img):
        height, width = cv_img.shape[:2]
        if len(cv_img.shape) == 2:
            bytes_per_line = width
            return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * width
            return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()

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
        return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()

class StereoVisionApp(QtWidgets.QMainWindow):
    MODES = [cv2.STEREO_SGBM_MODE_SGBM, cv2.STEREO_SGBM_MODE_HH, cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_HH4]
    MODE_NAMES = ["SGBM", "HH", "SGBM_3WAY", "HH4"]
    PREFILTER_TYPES = [cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE, cv2.STEREO_BM_PREFILTER_XSOBEL]
    PREFILTER_TYPE_NAMES = ["NORMALIZED_RESPONSE", "XSOBEL"]

    def __init__(self):
        super().__init__()

        self.point_cloud_window = PointCloudWindow()
        self.depth_map_window = DepthMapWindow()
        self.stereo_view_window = StereoViewWindow()

        self.stereo_view_window.show()

        self.current_index = 0
        self.display_mode = "RGB"
        self.depth_map_color = True
        self.use_sgbm = True 
        self.use_wls = False
        self.loadConfig()
        self.initUI()
        self.loadImagePaths()
        self.loadCameraParams()
        self.initData()
        self.image_resized = False
        self.updateDisparity()

    def loadConfig(self):
        with open("config/config.yml", "r") as file:
            self.config = yaml.safe_load(file)
        self.left_folder = self.config["left_folder"]
        self.right_folder = self.config["right_folder"]
        self.baseline = self.config["baseline"]

    def initUI(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QtWidgets.QHBoxLayout(central_widget)

        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
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
            "minDisparity": (0, 25, 0),
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
        self.slider_layout.addWidget(self.mode_label)
        self.slider_layout.addWidget(self.mode_slider)

        self.prefiltertype_label = QtWidgets.QLabel(f"preFilterType: {self.PREFILTER_TYPE_NAMES[0]}")
        self.prefiltertype_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.prefiltertype_slider.setMinimum(0)
        self.prefiltertype_slider.setMaximum(len(self.PREFILTER_TYPES) - 1)
        self.prefiltertype_slider.setValue(0)
        self.prefiltertype_slider.valueChanged.connect(self.updateDisparity)
        self.prefiltertype_slider.valueChanged.connect(lambda value: self.prefiltertype_label.setText(f"preFilterType: {self.PREFILTER_TYPE_NAMES[value]}"))
        self.slider_layout.addWidget(self.prefiltertype_label)
        self.slider_layout.addWidget(self.prefiltertype_slider)

        self.doffs_label = QtWidgets.QLabel("doffs: 0")
        self.doffs_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.doffs_slider.setMinimum(0)
        self.doffs_slider.setMaximum(100)
        self.doffs_slider.setValue(0)
        self.doffs_slider.valueChanged.connect(self.updateDisparity)
        self.doffs_slider.valueChanged.connect(lambda value: self.doffs_label.setText(f"doffs: {value}"))
        self.slider_layout.addWidget(self.doffs_label)
        self.slider_layout.addWidget(self.doffs_slider)

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

        self.display_button = QtWidgets.QPushButton()
        self.display_button.clicked.connect(self.toggleDisplayMode)
        self.button_layout.addWidget(self.display_button)

        self.depth_map_button = QtWidgets.QPushButton()
        self.depth_map_button.clicked.connect(self.toggleDepthMapColor)
        self.button_layout.addWidget(self.depth_map_button)

        self.toggle_algorithm_button = QtWidgets.QPushButton()
        self.toggle_algorithm_button.clicked.connect(self.toggleAlgorithm)
        self.button_layout.addWidget(self.toggle_algorithm_button)

        self.toggle_wls_button = QtWidgets.QPushButton()
        self.toggle_wls_button.clicked.connect(self.toggleWLS)
        self.button_layout.addWidget(self.toggle_wls_button)

        self.right_layout.addWidget(self.button_panel, 0, QtCore.Qt.AlignTop)

        self.depth_map_container = QtWidgets.QWidget()
        self.depth_map_layout = QtWidgets.QVBoxLayout(self.depth_map_container)
        self.depth_map_layout.setContentsMargins(0, 0, 0, 0)
        self.depth_map_layout.addWidget(self.depth_map_window, alignment=QtCore.Qt.AlignCenter)
        self.right_layout.addWidget(self.depth_map_container, 0, QtCore.Qt.AlignTop)

        self.right_layout.addWidget(self.point_cloud_window, 1)

        self.main_layout.addLayout(self.right_layout, 1)

        self.updateSliders()
        self.updateButtonLabels()

    def loadImagePaths(self):
        self.left_images = sorted([os.path.join(self.left_folder, img) for img in os.listdir(self.left_folder) if img.endswith('.png')])
        self.right_images = sorted([os.path.join(self.right_folder, img) for img in os.listdir(self.right_folder) if img.endswith('.png')])

        if len(self.left_images) != len(self.right_images):
            print("Error: The number of left and right images does not match.")
            exit()

    def loadImagePair(self):
        if 0 <= self.current_index < len(self.left_images):
            self.left_img = cv2.imread(self.left_images[self.current_index])
            self.right_img = cv2.imread(self.right_images[self.current_index])

            if self.left_img is None or self.right_img is None:
                print(f"Error: One or both images not found. Please check the file paths: {self.left_images[self.current_index]}, {self.right_images[self.current_index]}")
                exit()

            original_shape = self.left_img.shape
            self.left_img = self.resize_image(self.left_img, 600, 400)
            self.right_img = self.resize_image(self.right_img, 600, 400)

            if self.left_img.shape != original_shape:
                self.image_resized = True
            else:
                self.image_resized = False

    def resize_image(self, img, max_width, max_height):
        height, width = img.shape[:2]
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))
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
        self.loadImagePair()

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(self.K_left, self.dist_coeffs_left, self.K_right, self.dist_coeffs_right, self.left_img.shape[::-1][1:], self.R, self.T, alpha=0)
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(self.K_left, self.dist_coeffs_left, self.R1, self.P1, self.left_img.shape[1::-1], cv2.CV_32FC1)
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(self.K_right, self.dist_coeffs_right, self.R2, self.P2, self.right_img.shape[1::-1], cv2.CV_32FC1)

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
        self.loadImagePair()

        self.rectified_left = cv2.remap(self.left_img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        self.rectified_right = cv2.remap(self.right_img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

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

        # Display depth map
        if self.depth_map_color:
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map_normalized = np.uint8(depth_map_normalized)
            depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
            self.depth_map_window.updateDepthMap(depth_map_colored)
        else:
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map_normalized = np.uint8(depth_map_normalized)
            self.depth_map_window.updateDepthMap(depth_map_normalized)

        points = cv2.reprojectImageTo3D(disparity_map, self.Q)
        colors_bgr = self.rectified_left

        mask = disparity_map > min_disparity
        out_points = points[mask]

        if self.display_mode == "RGB":
            out_colors = colors_bgr[mask]
        elif self.display_mode == "BGR":
            out_colors = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)[mask]
        elif self.display_mode == "Greyscale":
            colors_gray = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2GRAY)
            out_colors = cv2.cvtColor(colors_gray, cv2.COLOR_GRAY2RGB)[mask]

        # There's a better way to do this but I'm lazy
        rotation_matrix_x = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]], dtype=np.float64)
        out_points = out_points.dot(rotation_matrix_x.T)

        rotation_matrix_y = np.array([[-1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, -1]], dtype=np.float64)
        out_points = out_points.dot(rotation_matrix_y.T)

        out_points[:, 2] = -out_points[:, 2]

        scaling_factor = 0.01
        out_points = out_points[::10] * scaling_factor
        out_colors = out_colors[::10] / 255.0

        self.point_cloud_window.updatePointCloud(out_points, out_colors)

        # Update rectified images
        if self.display_mode == "RGB":
            left_rectified_color = cv2.cvtColor(self.rectified_left, cv2.COLOR_BGR2RGB)
            right_rectified_color = cv2.cvtColor(self.rectified_right, cv2.COLOR_BGR2RGB)
        elif self.display_mode == "Greyscale":
            left_rectified_color = cv2.cvtColor(self.rectified_left, cv2.COLOR_BGR2GRAY)
            left_rectified_color = cv2.cvtColor(left_rectified_color, cv2.COLOR_GRAY2RGB)
            right_rectified_color = cv2.cvtColor(self.rectified_right, cv2.COLOR_BGR2GRAY)
            right_rectified_color = cv2.cvtColor(right_rectified_color, cv2.COLOR_GRAY2RGB)
        else:
            left_rectified_color = self.rectified_left
            right_rectified_color = self.rectified_right

        self.stereo_view_window.updateImages(left_rectified_color, right_rectified_color)

    def nextImage(self):
        if self.current_index < len(self.left_images) - 1:
            self.current_index += 1
            self.loadImagePair()
            self.updateDisparity()

    def prevImage(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.loadImagePair()
            self.updateDisparity()

    def updateButtonLabels(self):
        self.display_button.setText(f'Display Mode: {self.display_mode}')
        self.depth_map_button.setText(f'Depth Map Color: {"On" if self.depth_map_color else "Off"}')
        self.toggle_algorithm_button.setText(f'Algorithm: {"SGBM" if self.use_sgbm else "BM"}')
        self.toggle_wls_button.setText(f'WLS Filter: {"On" if self.use_wls else "Off"}')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = StereoVisionApp()
    mainWin.setGeometry(50, 50, 1200, 900) 
    mainWin.setWindowTitle('Stereo Vision Disparity Adjustment')
    mainWin.show()
    sys.exit(app.exec_())
