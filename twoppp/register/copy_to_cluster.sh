#!/bin/bash
shopt -s globstar

fly_dir=$1
to_warp='red_com_crop.tif'
ref_frame='ref_frame_com.tif'
warped='red_com_warped.tif'

convert_to_fidis_dir () {
    replace="scratch/jbraun"
    fidis_dir=${1/mnt\/nas\/JB/$replace}
    fidis_dir=${fidis_dir/mnt\/nas2\/JB/$replace}
    fidis_dir=${fidis_dir/mnt\/nas\/LH/$replace}
    fidis_dir=${fidis_dir/mnt\/nas2\/LH/$replace}
    fidis_dir=${fidis_dir/mnt\/data\/JB/$replace}
    fidis_dir=${fidis_dir/mnt\/data2\/JB/$replace}
    fidis_dir=${fidis_dir/mnt\/data\/LH/$replace}
    fidis_dir=${fidis_dir/mnt\/data2\/LH/$replace}
    echo ${fidis_dir}
}
convert_to_fidis_mounted_dir () {
    replace="scratch/jbraun"
    fidis_dir=${1/nas\/JB/$replace}
    fidis_dir=${fidis_dir/nas2\/JB/$replace}
    fidis_dir=${fidis_dir/nas\/LH/$replace}
    fidis_dir=${fidis_dir/nas2\/LH/$replace}
    fidis_dir=${fidis_dir/data\/JB/$replace}
    fidis_dir=${fidis_dir/data2\/JB/$replace}
    fidis_dir=${fidis_dir/data\/LH/$replace}
    fidis_dir=${fidis_dir/data2\/LH/$replace}
    echo ${fidis_dir}
}

copy_dir () {
    # fidis_dir=$(convert_to_fidis_dir ${1})
    fidis_dir=$(convert_to_fidis_mounted_dir ${1})
    fidis_base_dir=$(dirname ${fidis_dir})
    # echo MAKING DIR ${fidis_base_dir}
    mkdir -p ${fidis_base_dir}
    if test -f "${fidis_dir}"; then
        echo FILE "${fidis_dir}" ALREADY EXISTS
    else
        echo COPYING "${1}" TO "${fidis_dir}"
        cp "${1}" "${fidis_dir}"
    fi
    # rsync -az --rsync-path="mkdir -p ${fidis_base_dir} && rsync" ${1} jbraun@fdata1.epfl.ch:"$fidis_dir"
}
# echo FLY_DIR: "${fly_dir}"
# echo FIDIS_DIR: "$(convert_to_fidis_mounted_dir ${fly_dir})"
base_dir=$(dirname ${fly_dir})
# echo BASE DIR: "$base_dir"
if [[ $base_dir == *"fly"* ]] || [[ $base_dir == *"Fly"* ]]; then
    echo THIS IS A TRIAL DIR
    for dir in "$fly_dir/processed/$to_warp"; do
        # echo COPYING "${dir}"
        copy_dir ${dir}
    done
    for dir in "$fly_dir/processed/$warped"; do
        # echo COPYING "${dir}"
        copy_dir ${dir}
    done
else
    echo THIS IS A FLY DIR
    for dir in "$fly_dir/**/$ref_frame"; do
        # echo COPYING "${dir}"
        copy_dir ${dir}
    done
    for dir in $(find $fly_dir/**/**/$to_warp -type f); do  # $fly_dir/**/**/$to_warp"; do
        # echo FOUND "${dir}"
        copy_dir ${dir}
        # echo COPY TO "$(dirname ${dir})"
        # cp "$fly_dir/processed/$ref_frame" "$(dirname ${dir})"
    done
    for dir in $(find $fly_dir/**/**/$warped -type f); do  # $fly_dir/**/**/$to_warp"; do
        # echo FOUND "${dir}"
        copy_dir ${dir}
        # echo COPY TO "$(dirname ${dir})"
        # cp "$fly_dir/processed/$ref_frame" "$(dirname ${dir})"
    done
fi


