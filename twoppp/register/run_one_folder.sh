#!/bin/bash

folder=$1  # /scratch/jbraun/210723/fly1/cs_001/processed
time="16:00:00"
partition="parallel"  # "debug"  # "parallel"
output="./outputs/slurm-%j.out"
    
convert_to_fidis_dir () {
    fidis_dir=${1//mnt\/NAS\/JB/scratch\/jbraun}
    fidis_dir=${fidis_dir//mnt\/NAS2\/JB/scratch\/jbraun}
    fidis_dir=${fidis_dir//mnt\/NAS\/LH/scratch\/jbraun}
    fidis_dir=${fidis_dir//mnt\/NAS2\/LH/scratch\/jbraun}
    fidis_dir=${fidis_dir//mnt\/data\/JB/scratch\/jbraun}
    fidis_dir=${fidis_dir//mnt\/data2\/JB/scratch\/jbraun}
    fidis_dir=${fidis_dir//mnt\/data\/LH/scratch\/jbraun}
    fidis_dir=${fidis_dir//mnt\/data2\/LH/scratch\/jbraun}
    echo ${fidis_dir}
}

# while read dir; do 
if [[ -e ${folder} ]] && [[ ! -f "${folder}/red_com_warped.tif" ]] && [[ ! -f "${folder}/w.npy" ]]; then
    ref_frame="${folder}/../../processed/ref_frame_com.tif"
    echo ${folder}
    echo ${ref_frame}
    sbatch --nodes 1 --ntasks 1 --cpus-per-task 28 --time "${time}" --mem 128G --partition ${partition} --output "${output}" "registration_commands.sh" ${folder} ${ref_frame}
fi

# done <trials_to_warp.txt
