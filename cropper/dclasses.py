from dataclasses import dataclass
import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap


@dataclass
class Parameter:
    material: str = "Adamant"
    id: str = "NO-ID"
    location: str = "NO-ROOM"
    date: str = "1970-01-01"


@dataclass
class DataElement:
    ir_path: str = "res/missing.png"
    vis_path: str = "res/missing.png"
    raw_ir: np.array = None
    pm_ir: QPixmap = None
    pm_vis: QPixmap = None
    raw_vis: Image = None
    crop_ir: np.array = np.ones((30, 30))
    crop_vis: Image = Image.new('RGB', (100, 100), (255, 255, 255))
    raw_data: np.array = None
    target_path: str = "data/"
    info: Parameter = None
