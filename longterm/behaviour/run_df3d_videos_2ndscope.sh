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
   conda activate $TARGET
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
            # expand videos
            # vframes 100
            # https://unix.stackexchange.com/a/36363
            for ((i=0; i<7; i++)); do
                ffmpeg -i ""$folder"/camera_"$i".mp4" -qscale:v 2 -start_number 0 ""$folder"/camera_"$i"_img_%d.jpg"  < /dev/null
            done

            # run df3d
	        CUDA_VISIBLE_DEVICES=1 df3d-cli -vv -o $folder --output-folder df3d  # --camera-ids 6 5 4 3 2 1 0
            # for first scope: camera order 0, 6, 5, 4, 3, 2, 1

            delete images
            for ((i=0; i<7; i++)); do
                find "$folder" -name "*.jpg" -maxdepth 1  -delete
            done
        done
done < "$input"