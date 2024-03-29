#!/bin/bash

fly_dir=$1
to_warp='red_com_crop.tif'
ref_frame='ref_frame_com.tif'
warped='red_com_warped.tif'
weights="w.npy"

convert_to_fidis_dir () {
    echo "/scratch/$USER/${1#/mnt\/*\/*/}"
}
convert_to_fidis_mounted_dir () {
    echo "/mnt/scratch/$USER/${1#/mnt\/*\/*/}"
}

convert_to_file_serv_dir () {
    fidis_dir=${1}
    fly_dir=${2}
    fidis_fly_dir=${3}
    file_serv_dir=${fidis_dir/$fidis_fly_dir/$fly_dir}
    echo ${file_serv_dir}
}

copy_dir_reverse () {
    fidis_dir=${1}
    fly_dir=${2}
    fidis_fly_dir=$(convert_to_fidis_mounted_dir ${fly_dir})

    file_serv_dir=$(convert_to_file_serv_dir ${fidis_dir} ${fly_dir} ${fidis_fly_dir})
    if test -f "${file_serv_dir}"; then
        echo FILE "${file_serv_dir}" ALREADY EXISTS
    else
        echo COPYING "${fidis_dir}" TO "${file_serv_dir}"
        cp "${fidis_dir}" "${file_serv_dir}" 
    fi
    # rsync -az --rsync-path="mkdir -p ${fidis_base_dir} && rsync" ${1} jbraun@fdata1.epfl.ch:"$fidis_dir"
}
# echo FLY_DIR: "${fly_dir}"
# echo FIDIS_DIR: "$(convert_to_fidis_mounted_dir ${fly_dir})"
base_dir=$(dirname ${fly_dir})
# echo BASE DIR: "$base_dir"
fidis_fly_dir=$(convert_to_fidis_mounted_dir ${fly_dir})
echo ${fidis_fly_dir}
if [[ $base_dir == *"fly"* ]] || [[ $base_dir == *"Fly"* ]]; then
    echo THIS IS A TRIAL DIR
    for dir in "$fidis_fly_dir/processed/$warped"; do
        # echo COPYING "${dir}"
        copy_dir_reverse ${dir} ${fly_dir}
    done
    for dir in "$fidis_fly_dir/processed/$weights"; do
        # echo COPYING "${dir}"
        copy_dir_reverse ${dir} ${fly_dir}
    done
else
    echo THIS IS A FLY DIR
    for dir in $(find $fidis_fly_dir/**/**/$warped -type f); do
        # echo FOUND "${dir}"
        copy_dir_reverse ${dir} ${fly_dir}
    done
    for dir in $(find $fidis_fly_dir/**/**/$weights -type f); do
        # echo FOUND "${dir}"
        copy_dir_reverse ${dir} ${fly_dir}
    done
fi


