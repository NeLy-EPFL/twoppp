#!/bin/bash
time="50:00:00"
partition="debug"  # "parallel"
    
convert_to_fidis_dir () {
    fidis_dir=${1//mnt\/NAS\/JB/scratch\/jbraun}
    fidis_dir=${1//mnt\/NAS2\/JB/scratch\/jbraun}
    fidis_dir=${1//mnt\/NAS\/LH/scratch\/jbraun}
    fidis_dir=${1//mnt\/NAS2\/LH/scratch\/jbraun}
    fidis_dir=${1//mnt\/data\/JB/scratch\/jbraun}
    fidis_dir=${1//mnt\/data2\/JB/scratch\/jbraun}
    fidis_dir=${1//mnt\/data\/LH/scratch\/jbraun}
    fidis_dir=${1//mnt\/data2\/LH/scratch\/jbraun}
    echo ${fidis_dir}
}

# while read dir; do
for dir in $(find /scratch/jbraun/**/**/**/processed -type d); do
    echo ${dir}
    # folder=$(convert_to_fidis_dir ${dir})
    folder=${dir} 
    if [[ -e ${folder} ]] && [[ ! -f "${folder}/red_com_warped.tif" ]] && [[ ! -f "${folder}/w.npy" ]]; then
        ref_frame="${folder}/../../processed/ref_frame_com.tif"
        echo ${folder}
        echo ${ref_frame}
        echo sbatch --nodes 1 --ntasks 1 --cpus-per-task 28 --time "${time}" --mem 128G --partition ${partition} "registration_commands.sh" ${folder} ${ref_frame}
    fi
done
# done <trials_to_warp.txt