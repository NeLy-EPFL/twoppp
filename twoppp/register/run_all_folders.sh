#!/bin/bash
time="16:00:00"
partition="parallel"
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
for dir in $(find /scratch/jbraun/**/**/**/processed -type d); do
    echo ${dir}
    base_dir1=$(dirname ${dir})
    base_dir2=$(dirname ${base_dir1})
    if [[ $base_dir2 == *"fly"* ]] || [[ $base_dir2 == *"Fly"* ]]; then  # check whether path 2 folders up still contains fly
        # folder=$(convert_to_fidis_dir ${dir})
        folder=${dir} 
        if [[ -e ${folder} ]] && [[ ! -f "${folder}/red_com_warped.tif" ]]; then  #  && [[ ! -f "${folder}/w.npy" ]]; then
            ref_frame="${folder}/../../processed/ref_frame_com.tif"
            echo ${folder}
            echo ${ref_frame}
            sbatch --nodes 1 --ntasks 1 --cpus-per-task 28 --time "${time}" --mem 128G --partition ${partition} --output "${output}" "registration_commands.sh" ${folder} ${ref_frame}
        else
            echo "ALREADY PROCESSED: CONTINUE"
        fi
    else
        echo "THIS IS A FLY DIR: CONTINUE"
    fi
done
# done <trials_to_warp.txt
