#!/bin/bash

# make the conda command known to the current shell by performing source /.../conda.sh
BASE=$(conda info | grep -i 'base environment' | cut -d':' -f 2 | cut -d'(' -f 1 | cut -c2- | rev | cut -c3- | rev)
BASE="${BASE}/etc/profile.d/conda.sh"
echo "will source conda base directory: ${BASE}"
source $BASE

# activate the deepfly conda environment
TARGET="deepfly"
ENVS=$(conda env list | cut -d' ' -f 1 )
if [[ $ENVS = *"$TARGET"* ]]; then
   echo "Found environment. Will activate it."
   echo "previous environment: $CONDA_DEFAULT_ENV"
   conda activate $1
   echo "switched to: $CONDA_DEFAULT_ENV"
else 
   echo "Please create a conda environment called deepfly and install deepfly3d as specified here:"
   echo "https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/install.md"
exit
fi;

# perform deepfly3d on the specified folders
input="folders.txt"
while IFS= read -r folder_root
do
        find "$folder_root" -type d -name "images" -print0 | while read -d $'\0' folder
        do
            echo "$folder_root"
            echo $file

            # run df3d
	        CUDA_VISIBLE_DEVICES=1 df3d-cli -vv -o $folder --output-folder df3d  --camera-ids 6 5 4 3 2 1 0
            # for first scope: camera order 0, 6, 5, 4, 3, 2, 1

        done
done < "$input"