# twoppp: **two-p**hoton data **p**rocessing **p**ipeline

This package allows to process simulataneously recorded two-photon imaging data and behavioural data.

**!!!NEW!!!: checkout the new interface to automatically run the entire pipeline or subsets of it on multiple flies [here](twoppp/run/README.md):**

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

**NEW: checkout the new interface to automatically run the entire pipeline or subsets of it on multiple flies [here](twoppp/run/README.md):**

## Installation instructions:
### pure pip

- ```pip install numpy```
- ```pip install -e .```
- ```pip install numpy --upgrade```

### conda
Create conda environment and install twoppp package
- ```conda create -n twoppp37 python=3.7```
- ```conda activate twoppp37```
- ```pip install numpy```
- clone repository: ```git clone https://github.com/NeLy-EPFL/twoppp```
- change directory: ```cd twoppp```
- install using pip: ```pip install -e .```
- fix numpy installation: ```pip install numpy --upgrade```
- fix opencv installation:
   - ```pip uninstall opencv-contrib-python```
   - ```pip install opencv-contrib-python==4.2.0.32```
- fix pandas installation:
   - ```pip uninstall pandas```
   - ```pip install pandas==1.3.5```
