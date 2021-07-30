#!/bin/bash
shopt -s globstar

fly_dir=$1
to_warp='red_com_crop.tif'
warped='red_com_warped.tif'
ref_frame='ref_frame_com.tif'
weights="w.npy"

convert_to_fidis_dir () {
    replace="scratch/jbraun"
    fidis_dir=${1/mnt\/NAS\/JB/$replace}
    fidis_dir=${1/mnt\/NAS2\/JB/$replace}
    fidis_dir=${1/mnt\/NAS\/LH/$replace}
    fidis_dir=${1/mnt\/NAS2\/LH/$replace}
    fidis_dir=${1/mnt\/data\/JB/$replace}
    fidis_dir=${1/mnt\/data2\/JB/$replace}
    fidis_dir=${1/mnt\/data\/LH/$replace}
    fidis_dir=${1/mnt\/data2\/LH/$replace}
    echo ${fidis_dir}
}
convert_to_fidis_mounted_dir () {
    replace="scratch"
    fidis_dir=${1/NAS\/JB/$replace}
    fidis_dir=${1/NAS2\/JB/$replace}
    fidis_dir=${1/NAS\/LH/$replace}
    fidis_dir=${1/NAS2\/LH/$replace}
    # fidis_dir=${1/data\/JB/$replace}
    # fidis_dir=${1/data2\/JB/$replace}
    # fidis_dir=${1/data\/LH/$replace}
    # fidis_dir=${1/data2\/LH/$replace}
    echo ${fidis_dir}
}

convert_to_NAS2_dir () {
    replace="scratch"
    NAS2_dir=${1/$replace/NAS2\/LH}
    # fidis_dir=${1/NAS2\/JB/$replace}
    # fidis_dir=${1/NAS\/LH/$replace}
    # fidis_dir=${1/NAS2\/LH/$replace}
    # fidis_dir=${1/data\/JB/$replace}
    # fidis_dir=${1/data2\/JB/$replace}
    # fidis_dir=${1/data\/LH/$replace}
    # fidis_dir=${1/data2\/LH/$replace}
    echo ${NAS2_dir}
}

copy_dir_reverse () {
    fidis_dir=${1}
    NAS2_dir=$(convert_to_NAS2_dir ${1})
    echo COPYING "${fidis_dir}" TO "${NAS2_dir}"
    cp "${fidis_dir}" "${NAS2_dir}" 
    # rsync -az --rsync-path="mkdir -p ${fidis_base_dir} && rsync" ${1} jbraun@fdata1.epfl.ch:"$fidis_dir"
}
# echo FLY_DIR: "${fly_dir}"
# echo FIDIS_DIR: "$(convert_to_fidis_mounted_dir ${fly_dir})"
base_dir=$(dirname ${fly_dir})
# echo BASE DIR: "$base_dir"
fidis_fly_dir=$(convert_to_fidis_mounted_dir ${fly_dir})
echo ${fidis_fly_dir}
if [[ $base_dir == *"fly"* ]]; then
    echo THIS IS A TRIAL DIR
    for dir in "$fidis_fly_dir/processed/$warped"; do
        # echo COPYING "${dir}"
        copy_dir_reverse ${dir}
    done
    for dir in "$fidis_fly_dir/processed/$weights"; do
        # echo COPYING "${dir}"
        copy_dir_reverse ${dir}
    done
else
    echo THIS IS A FLY DIR
    for dir in $(find $fidis_fly_dir/**/**/$warped -type f); do
        # echo FOUND "${dir}"
        copy_dir_reverse ${dir}
    done
    for dir in $(find $fidis_fly_dir/**/**/$weights -type f); do
        # echo FOUND "${dir}"
        copy_dir_reverse ${dir}
    done
fi


