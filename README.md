# twoppp: **two-p**hoton data **p**rocessing **p**ipeline

This package allows to process simulataneously recorded two-photon imaging data and behavioural data.

**Most of the sub-modules and functions are already documented. More docstrings are forthcoming!**

This includes:
* two-photon processing:
    * converting from .raw to .tif
    * motion correction using center of mass and optical flow registration
        * both local and preparing for on cluster computation
    * denoising using DeppInterpolation
    * DF/F computation
    * ROI extraction
* behavioural data processing:
    * running DeepFly3D and postprocessing it
    * optical flow ball tracking processing
    * fictrac data handling
    * olfactory stimulation data processing
* combined behavioural and two-photon data processing:
    * synchronisation of the two (or three) modalities
    * creating pandas dataframes with synchronised data from behavioural and neural data
    * making videos

The "examples" folder showcases example usecases of each of the sub-modules.

If you want to run the entire processing pipeline in one go, look at the run_processing_pipeline_local.py script.

Installation instructions:
1. Create conda environment and install required packages according to their respective install instructions:
    - ```conda create -n twoppp37 python=3.7```
    - ```conda activate twoppp37```
    - ```conda install jupyter```
    - DeepFly3D: https://github.com/NeLy-EPFL/DeepFly3D (install in current entironment instead of making new one using ```pip install nely-df3d```)
    - deepinterpolation: https://github.com/NeLy-EPFL/deepinterpolation/tree/adapttoR57C10 (install in current environment instead of makeing a new environment)
        - this might throw an error like: "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed." ... for incompatibilities of numpy and scipy version, but this can be ignored.
    - ```conda install pandas```
    - ```conda install numpy```
    - utils2p: https://github.com/NeLy-EPFL/utils2p
    - ofco: https://github.com/NeLy-EPFL/ofco
        - this might again throw a dependency error about numpy and scipy, but it can be ignored.
    - utils_video: https://github.com/NeLy-EPFL/utils_video
    - df3dPostProcessing: https://github.com/NeLy-EPFL/df3dPostProcessing
        - use ```git clone https://github.com/NeLy-EPFL/df3dPostProcessing``` and ```pip install -e df3dPostProcessing``` if no install instructions are inside the repository
    - ```pip install behavelet```
    - try whether the pandas instruction works:
        - type ```python -c "import pandas as pd; print(pd.__version__)"```
        - If this returns just the version number, your're fine. If this returns an ImportError, this means that some of the installs messed up with the pandas/numpy dependencies. Fix the problem as follows:
        - ```pip uninstall pandas```
        - ```pip uninstall numpy```
        - ```conda install pandas```
2. Finally, we can install the twoppp package:
    - clone repository: ```git clone https://github.com/NeLy-EPFL/twoppp```
    - change directory: ```cd twoppp```
    - install using pip: ```pip install -e .```
        - if the install throws an error like "ImportError: cannot import name '_registerMatType' from 'cv2.cv2'", uninstall opencv and reinstall it. this is due to an opencv version incompatibility: ```pip uninstall opencv-python``` and ```pip install opencv-python```
