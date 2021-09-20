from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QPen, QBrush
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect

from PIL import Image, ImageOps, ImageEnhance
from PIL.ImageQt import ImageQt

from skimage.color import rgb2hsv

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import sys
from typing import List
from pathlib import Path
from glob import glob
from itertools import cycle
import json

import numpy as np

from cropper.dclasses import DataElement, Parameter


def get_params_from_name(file_name: str) -> dict:
    """
    Name Format: material_room_date_id.*
    Date Format: YYYY-MM-DD

    Args:
        file_name:

    Returns:
        All information stored in the filename string as dict.
    """
    # remove directory and file-ending
    parts = str(Path(file_name).stem)
    # split into data parts (every information is separated by a underscore)
    parts = parts.split("_")
    params = {
        "material": parts[0],
        "location": parts[1],
        "date": parts[2],
        "id": parts[3]
    }
    return params


def get_paths(data_path: str) -> List[DataElement]:
    """
    Loads all files from a specified directory as DataElement.

    Args:
        data_path: target directory with the infrared and visual files.

    Returns:
        list[DataElement]: A DataElement contains the paths to the ir and vis data, the material and a unique identifier
                           (the filename).
    """
    ir_files = sorted(glob(data_path+"/*.csv"))
    vis_files = sorted(glob(data_path+"/*.jpg"))
    data: List[DataElement] = []

    for i in range(len(ir_files)):
        element = DataElement()
        element.ir_path = str(Path(ir_files[i]))
        element.vis_path = str(Path(vis_files[i]))
        element.info = Parameter(**(get_params_from_name(element.ir_path)))
        data.append(element)

    return data


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, width=5, height=4, dpi=100):
        font = {'size': 10}
        matplotlib.rc('font', **font)
        self.mode = "Hue"
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title(f"{self.mode} Histogram")
        self.fig.tight_layout(pad=1)
        super(MplCanvas, self).__init__(self.fig)

    def update_plot(self, x: np.array, y: np.array, mode):
        self.axes.cla()
        self.axes.plot(x, y)
        self.axes.set_title(f"{mode} Histogram")
        self.draw()


class GUI(QWidget):

    def __init__(self):
        super().__init__()

        # pick data directory
        directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.data = get_paths(directory)
        self.setWindowIcon(QIcon(str(Path('res/icon.png'))))
        self.config: dict = self.load_settings(str(Path("res/settings.json")))
        self.settings: dict = self.config["settings"]
        self.tooltip_dict: dict = self.config["tooltips"]

        self.idx = 0
        self.currentIR = None
        self.currentVis = None
        self.vis_pos = [0, 0]
        self.ir_pos = [0, 0]

        # colors
        self.blue70 = QtGui.QColor(0, 0, 255, 179)
        self.red70 = QtGui.QColor(255, 0, 0, 179)

        # T_u calculation
        self.temp_pos = [0, 0]
        self.temp_size = self.settings["temp_area"]
        self.temp_mean = 0.0
        self.show_temp_rect = True

        # resolution scaling factor
        # (the visual data is twice the size (resolution) compared to the ir data)
        self.ir_scale = 1.0  # 640 / 640
        self.vis_scale = 2.0  # 1280 / 640
        self.crop_display_size = 300

        # size of the cropping rectangle
        self.crop_size_vis = self.settings["crop_size_vis"]
        self.crop_size_ir = self.settings["crop_size_ir"]

        # toggle operation applied to infrared image
        self.lst_op = ["Equalize", "Contrast", "None"]
        self.av_operations = cycle(self.lst_op)
        self.operation = next(self.av_operations)
        self.contrast = self.settings["contrast"]

        # visual histogram
        self.sc = MplCanvas(width=5, height=4)
        self.hist_bins = self.settings["hist_bins"]

        # layout
        self.grid = QGridLayout()

        # visual image
        self.vis_label = QLabel()
        self.vis_label.mousePressEvent = self.interact_visual

        # infrared image
        self.ir_label = QLabel()
        self.ir_label.mousePressEvent = self.interact_infrared

        # material name
        self.material_label = QLabel()

        # index counter
        self.idx_label = QLabel()

        # crop labels
        self.crop_vis_label = QLabel()
        self.crop_ir_label = QLabel()

        # mouse position labels
        self.vis_pos_label = QLabel()
        self.vis_pos_label.setText("0, 0")
        self.ir_pos_label = QLabel()
        self.ir_pos_label.setText("0, 0")

        # T_u Label
        self.temp_label = QLabel()
        self.temp_label.setText(f"T_u: {self.temp_mean:.5}")

        # control buttons
        self.nextBtn = QPushButton("Next")
        self.nextBtn.clicked.connect(self.next_image)
        self.prevBtn = QPushButton("Previous")
        self.prevBtn.clicked.connect(self.prev_image)
        self.saveBtn = QPushButton("Save")
        self.saveBtn.clicked.connect(self.save_cropped_data)
        self.toggleBtn = QPushButton("Equalize")
        self.toggleBtn.clicked.connect(self.toggle_operation)
        self.toggleBtn.setToolTip(self.tooltip_dict["Equalize"])

        # current Data
        if len(self.data) > 0:
            self.cData: DataElement = self.data[self.idx]
            self.update_data()

        else:
            print("Folder is empty!")
            sys.exit(1)

        self.setMouseTracking(True)
        self.init_ui()

    @staticmethod
    def load_settings(path: str):
        try:
            with open(path) as f:
                settings = json.load(f)
            return settings
        except FileNotFoundError:
            print("Settings file is missing!")

    def init_ui(self):
        """
        Initializes the PyQT User Interface. Sets the default values to all labels. 
        
        Returns:
            Nothing.
        """
        # aligns index label to the right
        self.idx_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # first row | material and index
        self.grid.addWidget(self.material_label, 1, 1, 1, 2)
        self.grid.addWidget(self.idx_label, 1, 3, 1, 2)

        # second row | visual and infrared data as image
        self.grid.addWidget(self.vis_label, 2, 1, 1, 2)  # vis image
        self.grid.addWidget(self.ir_label, 2, 3, 1, 2)  # ir "image"

        # third row | cropped visual and infrared data as image
        self.grid.addWidget(self.crop_vis_label, 3, 1)  # cropped vis
        self.grid.addWidget(self.sc, 3, 2)
        self.grid.addWidget(self.crop_ir_label, 3, 3)  # cropped ir
        self.grid.addWidget(self.temp_label, 3, 4, alignment=Qt.AlignLeft)

        # fourth row | mouse position for cropping
        self.grid.addWidget(self.vis_pos_label, 4, 1)  # vis
        self.grid.addWidget(self.ir_pos_label, 4, 3)  # ir

        # fifth row | control buttons
        self.grid.addWidget(self.prevBtn, 5, 1)  # previous
        self.grid.addWidget(self.saveBtn, 5, 2)  # save
        self.grid.addWidget(self.toggleBtn, 5, 3)  # equalize
        self.grid.addWidget(self.nextBtn, 5, 4)  # next

        # apply layout
        self.setLayout(self.grid)

        # set window size
        self.setGeometry(100, 100, 300, 300)

        # set title
        self.setWindowTitle("Cropping Tool")
        self.show()

    def update_data(self):
        """
        Loads next data pair (visual and infrared) and updates all labels. Also draws the rectangles on their current
        position.
        """
        # set up infrared
        # resolution: width: 640, height: 480
        self.cData.raw_ir = np.genfromtxt(self.cData.ir_path, delimiter=",")
        ir_img = self.image_from_float(self.cData.raw_ir)
        self.currentIR = QPixmap()
        self.currentIR = QPixmap.fromImage(ir_img)
        self.cData.pm_ir = self.currentIR

        # update T_u and ir crop label
        self.update_mean_temp()
        self.crop_ir()

        # set labels
        self.ir_label.setPixmap(self.currentIR)
        self.update_ir_label(self.cData.crop_ir)

        # set up visual
        self.cData.raw_vis = Image.open(self.cData.vis_path)
        self.currentVis = QPixmap(self.cData.vis_path)
        self.currentVis = self.currentVis.scaledToWidth(640)
        self.cData.pm_vis = self.currentVis

        # update cropped visual (data)
        self.crop_vis()
        self.draw_vis_crop_rect()

        # set labels
        self.vis_label.setPixmap(self.currentVis)
        self.update_vis_label(self.cData.crop_vis)

        # set up material label
        self.material_label.setText(self.cData.info.material)

        # set up idx counter
        self.idx_label.setText(f"{self.idx+1}/{len(self.data)}")

        # setup toggle button
        self.toggleBtn.setText(self.operation)
        self.toggleBtn.setToolTip(self.tooltip_dict[self.operation])

    def create_histogram(self, img: Image):
        hsv_img = rgb2hsv(np.array(img))
        hue_channel = hsv_img[:, :, 0]
        # sat_channel = hsv_img[:, :, 1]
        # val_channel = hsv_img[:, :, 2]

        data = np.array(hue_channel).flatten()
        hist_data, bins, patches = plt.hist(data, bins=self.hist_bins, weights=np.ones_like(data) / data.size)
        self.sc.update_plot([i+1 for i in range(self.hist_bins)], hist_data, "Hue")

    def image_from_float(self, data: np.array) -> QImage:
        """
        Converts a numpy array with float values to a displayable image. Applies histogram equalization.

        Args:
            arr [np.array]: data for conversion

        Returns:
            QImage: Image Format for PyQt applications.
        """
        # convert data to integer values
        arr = data.copy()
        arr = arr.astype(np.float64)
        arr = arr / np.max(arr)
        arr = 255 * arr
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)

        # histogram equalization
        if self.operation == "Equalize":
            img = ImageOps.equalize(img, mask=None)

        # increasing contrast
        elif self.operation == "Contrast":
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)

        img = img.convert("RGBA")

        # convert to PyQt Image
        img = ImageQt(img)
        return img

    def crop_vis(self):
        # correct resolution
        x = int(self.vis_pos[0]*2)
        y = int(self.vis_pos[1]*2)

        # load image
        img = self.cData.raw_vis
        width, height = img.size

        # check out of bounds
        if x+self.crop_size_vis > width:
            x = int(width - self.crop_size_vis)

        if y + self.crop_size_vis > height:
            y = int(height - self.crop_size_vis)

        # crop image
        crop_img = img.crop((x, y, x+self.crop_size_vis, y+self.crop_size_vis))
        self.cData.crop_vis = crop_img

        self.create_histogram(crop_img.copy())

        self.update_vis_label(self.cData.crop_vis)

    def get_temp_area(self, pos: np.array, size: int):
        """
        Extracts a specific area of the temperature array.

        Args:
            pos: top left corner of the area rectangle.
            size: width and height of the rectangle.

        Returns:
            cropped_data: the desired data.
            new_pos: if the rectangle is out of bounds the coordinates of the top left corner are reset
                     to the next possible location. This variable contains the changed position.
        """
        # get data
        arr: np.array = self.cData.raw_ir

        # correct resolution (per default the correction factor is 1.0)
        x = int(self.ir_scale * pos[0])
        y = int(self.ir_scale * pos[1])

        # check out of bounds
        if x+size > arr.shape[1]:
            x = int(arr.shape[1] - size)

        if y + size > arr.shape[0]:
            y = int(arr.shape[0] - size)

        # update pos (fix out of bounds)
        new_pos = [x, y]

        # get temperature area
        cropped_data = arr[y:y+int(size), x:x+int(size)]
        return cropped_data, new_pos

    def crop_ir(self):
        cropped_data, self.ir_pos = self.get_temp_area(self.ir_pos, self.crop_size_ir)
        self.cData.crop_ir = cropped_data
        self.update_ir_label(cropped_data)
        self.draw_ir_rect()

    def update_ir_label(self, cropped_data: np.array):
        """
        Updates the label for the cropped infrared image.

        Args:
            cropped_data [np.array]:

        Returns:
            Nothing.
        """
        ir_img = self.image_from_float(cropped_data)
        ir_pm = QPixmap.fromImage(ir_img)
        ir_pm = ir_pm.scaledToWidth(self.crop_display_size)
        self.crop_ir_label.setPixmap(ir_pm)

    def update_vis_label(self, crop_img: Image):
        """
        Updates the label for the cropped visual image.

        Args:
            crop_img:

        Returns:
            Nothing
        """
        vis_img = ImageQt(crop_img)
        vis_pm = QPixmap.fromImage(vis_img)
        vis_pm = vis_pm.scaledToWidth(self.crop_display_size)
        self.crop_vis_label.setPixmap(vis_pm)

    def update_mean_temp(self):
        """
        Calculates the mean temperature in a specific area (self.temp_pos, self.temp_size).
        Stores the calculated temperature in self.temp_mean.

        Returns:
            Nothing.
        """
        temp_data, self.temp_pos = self.get_temp_area(self.temp_pos, self.temp_size)
        if self.show_temp_rect:
            self.draw_ir_rect()
        self.temp_mean = temp_data.mean()
        self.temp_label.setText(f"T_u: {self.temp_mean:.5}")

    ###################
    # draw rectangles #
    ###################
    def draw_vis_crop_rect(self):
        width, height = self.vis_label.width(), self.vis_label.height()
        size = int(self.crop_size_vis//self.vis_scale)

        if self.vis_pos[0] + size > width:
            self.vis_pos[0] = int(width - size)

        if self.vis_pos[1] + size > height:
            self.vis_pos[1] = int(height - size)

        self.currentVis = self.cData.pm_vis.copy()

        painter = QPainter(self.currentVis)
        painter.setPen(QPen(Qt.blue, 3, Qt.SolidLine))
        painter.setBrush(QBrush(self.blue70, Qt.DiagCrossPattern))
        rect = QRect(self.vis_pos[0], self.vis_pos[1], size, size)

        painter.drawRect(rect)
        size = int(size*self.vis_scale)
        painter.drawText(rect, Qt.AlignCenter, "{}\nx\n{}".format(size, size))

        self.vis_label.setPixmap(self.currentVis)

    def draw_ir_rect(self):
        """
        Draws two rectangles on the infrared label. The blue rectangle represents the captured area, which is used to
        store the desired temperature data for the specific material. The red rectangle captures the area which is used
        to approximate an ambient temperature, by calculating the mean.

        Returns:
            Nothing.
        """
        self.currentIR = self.cData.pm_ir.copy()
        size = int(self.crop_size_ir // self.ir_scale)

        crop_rect = QRect(self.ir_pos[0], self.ir_pos[1], size, size)
        temp_rect = QRect(self.temp_pos[0], self.temp_pos[1], self.temp_size, self.temp_size)
        painter = QPainter(self.currentIR)

        # draw crop rectangle
        painter.setPen(QPen(Qt.blue, 3, Qt.SolidLine))
        painter.setBrush(QBrush(self.blue70, Qt.DiagCrossPattern))
        painter.drawRect(crop_rect)
        painter.drawText(crop_rect, Qt.AlignCenter, f"{size}x{size}")

        # draw T_u rectangle
        painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
        painter.setBrush(QBrush(self.red70, Qt.DiagCrossPattern))
        painter.drawRect(temp_rect)

        self.ir_label.setPixmap(self.currentIR)

    #####################
    # user interactions #
    #####################
    def next_image(self):
        """
        Cycles to the next image. When the last image is reached, the counter is reset to the first one.

        Returns:
            Nothing.
        """
        if len(self.data) > self.idx+1:
            self.idx += 1
        else:
            print("All images processed!")
            self.idx = 0

        self.cData = self.data[self.idx]
        self.update_data()

    def prev_image(self):
        """
        Cycles to the previous image. When the first image is reached, the counter is set to the last one.

        Return:
            Nothing.
        """
        if self.idx > 0:
            self.idx -= 1
        else:
            self.idx = len(self.data)-1

        self.cData = self.data[self.idx]
        self.update_data()

    def save_cropped_data(self):
        """
        Save the currently targeted areas in the visual and infrared data. The visual data is stored as .png
        and the infrared data as .json-File. Besides the temperature array the determined ambient temperature,
        the date of the data recording and the location of the recording is stored.

        Returns:
            Nothing.
        """
        dir_path = Path(f"{self.cData.target_path}/{self.cData.info.material}")
        dir_path.mkdir(exist_ok=True, parents=True)
        i = 0
        vis_path = Path(str(dir_path) + f"/{self.cData.info.location}_"
                                        f"{self.cData.info.date}_"
                                        f"{self.cData.info.id}-{i}.png")
        ir_path = Path(str(dir_path) + f"/{self.cData.info.location}_"
                                       f"{self.cData.info.date}_"
                                       f"{self.cData.info.id}-{i}.json")

        while vis_path.is_file():
            i += 1
            vis_path = Path(str(dir_path)+f"/{self.cData.info.location}_"
                                          f"{self.cData.info.date}_"
                                          f"{self.cData.info.id}-{i}.png")
            ir_path = Path(str(dir_path)+f"/{self.cData.info.location}_"
                                         f"{self.cData.info.date}_"
                                         f"{self.cData.info.id}-{i}.json")

        # save visual data as png
        self.cData.crop_vis.save(vis_path)

        # save infrared data as json
        ir_data = {"thermal": self.cData.crop_ir.tolist(),
                   "temp": self.temp_mean,
                   "material": self.cData.info.material,
                   "room": self.cData.info.location}

        with open(ir_path, 'w+') as f:
            content = json.dumps(ir_data)
            f.write(content)

    def interact_infrared(self, event):
        """
        Interaction with the infrared label. Left-click sets the top left corner for the capturing rectangle.
        Right-click sets the top left corner for the temperature rectangle. The latter is used to approximate the
        ambient temperature (mean).

        Args:
            event: PyQt Mouseevent

        Returns:
            Nothing.
        """
        # crop ir image
        if event.button() == Qt.LeftButton:
            self.ir_pos = [event.x(), event.y()]
            self.ir_pos_label.setText(f"{event.x()}, {event.y()}")
            self.crop_ir()

        # get temperature
        elif event.button() == Qt.RightButton:
            self.temp_pos = [event.x(), event.y()]
            self.update_mean_temp()

    def interact_visual(self, event):
        """
        Interaction with the visual label. Left-click sets the top left corner of the capturing rectangle.

        Args:
            event: PyQt Mouseevent

        Returns:
            Nothing.
        """
        # crop visual image
        if event.button() == Qt.LeftButton:
            self.vis_pos = [event.x(), event.y()]
            self.vis_pos_label.setText(f"{event.x()}, {event.y()}")
            self.crop_vis()
            self.draw_vis_crop_rect()

    def toggle_operation(self):
        """
        Cycles through the different filter types for the infrared image.

        Returns:
            Nothing.
        """
        self.operation = next(self.av_operations)
        self.update_data()
