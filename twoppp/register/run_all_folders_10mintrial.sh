#!/bin/bash
time="40:00:00"
partition="parallel"
output="./outputs/slurm-%j.out"    
mkdir -p ./outputs

# while read dir; do
for dir in $(find /scratch/$USER/**/**/**/processed -type d); do
    echo ${dir}
    base_dir1=$(dirname ${dir})
    base_dir2=$(dirname ${base_dir1})
    if [[ $base_dir2 == *"fly"* ]] || [[ $base_dir2 == *"Fly"* ]]; then  # check whether path 2 folders up still contains fly
        folder=${dir} 
        if [[ -e ${folder} ]] && [[ ! -f "${folder}/red_com_warped.tif" ]]; then  #  && [[ ! -f "${folder}/w.npy" ]]; then
            ref_frame="${folder}/../../processed/ref_frame_com.tif"
            echo ${folder}
            echo ${ref_frame}
            sbatch --nodes 1 --ntasks 1 --cpus-per-task 28 --time "${time}" --mem 128G --partition ${partition} --output "${output}" warping_cluster.py ${folder} ${ref_frame}
        else
            echo "ALREADY PROCESSED: CONTINUE"
        fi
    else
        echo "THIS IS A FLY DIR: CONTINUE"
    fi
done
# done <trials_to_warp.txt
