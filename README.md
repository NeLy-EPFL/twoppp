# twoppp: **two-p**hoton data **p**rocessing **p**ipeline

This package allows to process simulataneously recorded two-photon imaging data and behavioural data.

**Examples to document basic uses are forthcoming!**

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

Installation instructions:
1. Create conda environment and install required packages according to their respective install instructions:
    - ```conda create -n twoppp37 python=3.7```
    - ```conda activate twoppp37```
    - ```conda install jupyter```
    - DeepFly3D: https://github.com/NeLy-EPFL/DeepFly3D
    - deepinterpolation: https://github.com/NeLy-EPFL/deepinterpolation/tree/adapttoR57C10
        - this might throw an error like: "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed." ... for incompatibilities of numpy and scipy version, but this can be ignored.
    - ```conda install pandas```
    - ```conda install numpy```
    - utils2p: https://github.com/NeLy-EPFL/utils2p
    - ofco: https://github.com/NeLy-EPFL/ofco
        - this might again throw a dependency error about numpy and scipy, but it can be ignored.
    - utils_video: https://github.com/NeLy-EPFL/utils_video
    - df3dPostProcessing: https://github.com/NeLy-EPFL/df3dPostProcessing
    - ```pip install behavelet```
2. clone repository: ```git clone https://github.com/NeLy-EPFL/twoppp```
3. change directory: ```cd twoppp```
3. install using pip: ```pip install -e .```
