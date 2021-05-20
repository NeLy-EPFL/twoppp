import os, sys

import deepinterpolation.interface as denoise

di_data_dir = "/home/jbraun/tmp/deepinterpolation/data"
di_run_dir = "/home/jbraun/tmp/deepinterpolation/runs/210427_fly1_cs_002"

raw_data_tifs = ['/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_002/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_caff/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_caff_after/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_caff_after_006/processed/green_com_warped.tif']

tmp_data_tifs = [di_data_dir + "/210427_fly1_tmp_{}.tif".format(i) for i in range(4)]

denoised_data_tifs = ['/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_002/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_caff/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_caff_after/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210427_caffeine/J1M5_fly1/cs_caff_after_006/processed/green_denoised_t1.tif']

# denoise.prepare_data(raw_data_tifs, tmp_data_tifs, offset=(None, None), size=(320,448))
# denoise.inference(tmp_data_tifs, run_dir=di_run_dir, tif_out_dirs=denoised_data_tifs)

### 210423
di_run_dir = "/home/jbraun/tmp/deepinterpolation/runs/210423_fly2_cs_001"

raw_data_tifs = ['/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_001/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_caff/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_caff_after/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_caff_after_006/processed/green_com_warped.tif']

tmp_data_tifs = [di_data_dir + "/210423_fly2_tmp_{}.tif".format(i) for i in range(4)]

denoised_data_tifs = ['/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_001/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_caff/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_caff_after/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210423_caffeine/J1M5_fly2/cs_caff_after_006/processed/green_denoised_t1.tif']

# denoise.prepare_data(raw_data_tifs, tmp_data_tifs, offset=(None, None), size=(320,448))
# denoise.inference(tmp_data_tifs, run_dir=di_run_dir, tif_out_dirs=denoised_data_tifs)


### 210415
di_run_dir = "/home/jbraun/tmp/deepinterpolation/runs/210415_fly2_cs_003"

raw_data_tifs = ['/mnt/NAS/LH/210415/J1M5_fly2/cs_003/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210415/J1M5_fly2/cs_water/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210415/J1M5_fly2/cs_water_after_001/processed/green_com_warped.tif',
                 '/mnt/NAS/LH/210415/J1M5_fly2/cs_water_after_003/processed/green_com_warped.tif']

tmp_data_tifs = [di_data_dir + "/210415_fly2_tmp_{}.tif".format(i) for i in range(4)]

denoised_data_tifs = ['/mnt/NAS/LH/210415/J1M5_fly2/cs_003/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210415/J1M5_fly2/cs_water/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210415/J1M5_fly2/cs_water_after_001/processed/green_denoised_t1.tif',
                      '/mnt/NAS/LH/210415/J1M5_fly2/cs_water_after_003/processed/green_denoised_t1.tif']

# denoise.prepare_data(raw_data_tifs, tmp_data_tifs, offset=(None, None), size=(320,640))
# denoise.inference(tmp_data_tifs, run_dir=di_run_dir, tif_out_dirs=denoised_data_tifs)

### 210512
di_run_dir = "/home/jbraun/tmp/deepinterpolation/runs/210512_fly3_cs_002"

raw_data_tifs = [# '/mnt/NAS2/LH/210512/fly3/cs_002/processed/green_com_warped.tif',
                 '/mnt/NAS2/LH/210512/fly3/cs_caff/processed/green_com_warped.tif',
                 '/mnt/NAS2/LH/210512/fly3/cs_caff_after/processed/green_com_warped.tif',
                 '/mnt/NAS2/LH/210512/fly3/cs_caff_after_006/processed/green_com_warped.tif']

tmp_data_tifs = [di_data_dir + "/210512_fly3_tmp_{}.tif".format(i) for i in range(1,4)]

denoised_data_tifs = [# '/mnt/NAS2/LH/210512/fly3/cs_002/processed/green_denoised_t1.tif',
                      '/mnt/NAS2/LH/210512/fly3/cs_caff/processed/green_denoised_t1.tif',
                      '/mnt/NAS2/LH/210512/fly3/cs_caff_after/processed/green_denoised_t1.tif',
                      '/mnt/NAS2/LH/210512/fly3/cs_caff_after_006/processed/green_denoised_t1.tif']

# denoise.prepare_data(raw_data_tifs, tmp_data_tifs, offset=(None, None), size=(352,640))
# denoise.inference(tmp_data_tifs, run_dir=di_run_dir, tif_out_dirs=denoised_data_tifs)

### 210514
di_run_dir = "/home/jbraun/tmp/deepinterpolation/runs/210514_fly1_cs"

raw_data_tifs = [# '/mnt/NAS2/LH/210514/fly1/cs/processed/green_com_warped.tif',
                 '/mnt/NAS2/LH/210514/fly1/cs_caff_001/processed/green_com_warped.tif',
                 '/mnt/NAS2/LH/210514/fly1/cs_caff_after/processed/green_com_warped.tif',
                 '/mnt/NAS2/LH/210514/fly1/cs_caff_after_006/processed/green_com_warped.tif']

tmp_data_tifs = [di_data_dir + "/210514_fly1_tmp_{}.tif".format(i) for i in range(1,4)]

denoised_data_tifs = [# '/mnt/NAS2/LH/210514/fly1/cs/processed/green_denoised_t1.tif',
                      '/mnt/NAS2/LH/210514/fly1/cs_caff_001/processed/green_denoised_t1.tif',
                      '/mnt/NAS2/LH/210514/fly1/cs_caff_after/processed/green_denoised_t1.tif',
                      '/mnt/NAS2/LH/210514/fly1/cs_caff_after_006/processed/green_denoised_t1.tif']

denoise.prepare_data(raw_data_tifs, tmp_data_tifs, offset=(None, None), size=(352,640))
denoise.inference(tmp_data_tifs, run_dir=di_run_dir, tif_out_dirs=denoised_data_tifs)

