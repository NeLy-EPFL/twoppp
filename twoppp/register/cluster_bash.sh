#!/bin/bash
time="50:00:00"
partition="debug"  # "parallel"
output="./outputs/slurm-%j.out"

# while read dir; do
for dir in $(find /scratch/jbraun/**/**/**/processed -type d); do
    echo ${dir}
    folder=${dir} 
    if [[ -e ${folder} ]] && [[ ! -f "${folder}/red_com_warped.tif" ]] && [[ ! -f "${folder}/w.npy" ]]; then
        ref_frame="${folder}/../../processed/ref_frame_com.tif"
        echo ${folder}
        echo ${ref_frame}
        echo sbatch --nodes 1 --ntasks 1 --cpus-per-task 28 --time "${time}" --mem 128G --partition ${partition} --output "${output}" "registration_commands.sh" ${folder} ${ref_frame}
    fi
done
# done <trials_to_warp.txt