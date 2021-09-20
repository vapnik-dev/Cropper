from cropper import core

from PyQt5.QtWidgets import QApplication
import sys
import ctypes
import os


if __name__ == "__main__":

    # check if running on windows
    if os.name == "nt":
        app_id = 'imb.cropper'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

    # run application
    app = QApplication(sys.argv)
    ex = core.GUI()
    sys.exit(app.exec_())
