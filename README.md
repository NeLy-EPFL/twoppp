# twoppp: **two-p**hoton data **p**rocessing **p**ipeline

This package allows to process simulataneously recorded two-photon imaging data and behavioural data.\n
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
    * making videos

Installation instructions:
1. Create conda environment and install required packages according to their respective install instructions:
    - ```conda create -n twoppp37 python=3.7```
    - ```conda activate twoppp37```
    - ```conda install jupyter```
    - DeepFly3D: https://github.com/NeLy-EPFL/DeepFly3D
    - deepinterpolation: https://github.com/NeLy-EPFL/deepinterpolation/tree/adapttoR57C10
    - ```conda install pandas```
    - ```conda install numpy```
    - utils2p: https://github.com/NeLy-EPFL/utils2p
    - ofco: https://github.com/NeLy-EPFL/ofco
    - utils_video: https://github.com/NeLy-EPFL/utils_video
    - df3dPostProcessing: https://github.com/NeLy-EPFL/df3dPostProcessing
    - ```conda install os sys pathlib shutil numpy array gc copy tqdm sklearn pickle glob matplotlib math cv2 json pandas scipy```
2. clone repository: ```git clone https://github.com/NeLy-EPFL/twoppp```
3. run setup.py: ```python twoppp/.setup.py```
