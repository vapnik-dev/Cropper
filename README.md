### Cropping Tool for infrared and visual images
This tool is designed to capture specific areas of visual (VIS) and infrared (IR) images to extract 
information about the electromagnetic emissions of different materials. 

### Quickstart Guide
**1. Install Python (minimum version 3.8)**
   1. *Windows* \
   Download [Python](https://www.python.org/downloads/) and follow the instructions of the installer.
   2. *Linux* \
   Python should be pre-installed in common Linux distributions. To check your version number open a terminal and type
   python3 --version. 

**2. Create a virtual environment and activate it**
   1. *Windows* \
   Open a Powershell at the location you want to create the environment. \
   (**Right-click+shift** -> **Open Powershell**) \
   Run `python -m venv NAME` replace **NAME** with a name for your environment (something like "venv") \
   To activate the environment you have to run `NAME/bin/Scripts/Activate`
   2. *Linux* \
   Open a Terminal at  the location you want to create the environment. \
   (**Right-click** -> **Open Terminal here**) \
   Run `python -m venv NAME` replace **NAME** with a name for your environment (something like "venv") \
   To activate the environment you have to run `source NAME/bin/activate` 

**3. Install all necessary packages** \
   `python -m pip install -r requirements.txt` 

**4. Run the Program** \
   `python run.py`

### Settings
It is possible to tweak some variables in the settings-file which is located in the `res` directory.\
`temp_area` [int]: Adjusts the size of the rectangle which approximates the ambient temperature. \
`crop_size_ir` [int]: Adjusts the size of the rectangle which captures the IR-data. \
`crop_size_vis` [int]: Adjusts the size of the rectangle which captures the VIS-data. \
`hist_bins` [int]: Adjusts the number of different hue values which are displayed in the Hue-histogram. \
`contrast` [int]: Adjusts the value for the contrast correction. The high values lead to high contrast.