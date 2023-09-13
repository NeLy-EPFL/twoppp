# twoppp: **two-p**hoton data **p**rocessing **p**ipeline

This package allows to process simulataneously recorded two-photon imaging data and behavioural data.

**The standard interface to automatically run the entire pipeline or subsets of it on multiple flies is documented [here](twoppp/run/README.md):**

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
    * running Sleap for 2D pose estimation and processing it
    * optical flow ball tracking processing
    * fictrac data handling
    * wheel speed calculation
    * olfactory stimulation data processing
    * optogenetic stimulus processing
* combined behavioural and two-photon data processing:
    * synchronisation of the two (or three) modalities
    * creating pandas dataframes with synchronised data from behavioural and neural data
    * making videos

The "examples" folder showcases example usecases of each of the sub-modules.

**The new (2022) interface to automatically run the entire pipeline or subsets of it on multiple flies is documented [here](twoppp/run/README.md):**

## Installation instructions:

### conda (preferred)

Create conda environment and install twoppp package
- ```conda create -n twoppp37 python=3.7```
- ```conda activate twoppp37```
- ```pip install numpy```
- clone repository: ```git clone https://github.com/NeLy-EPFL/twoppp```
- change directory: ```cd twoppp```
- install using pip: ```pip install -e .```
- fix numpy installation: ```pip install numpy --upgrade``` (ignore the warning message about numpy incompatibility)

### pure pip

- ```pip install numpy```
- ```pip install -e .```
- ```pip install numpy --upgrade```
