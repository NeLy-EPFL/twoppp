#!/bin/sh

# largely copied and slightly modified from 
# https://github.com/NeLy-EPFL/ofco/blob/master/examples/registration_commands.sh
folder=$1
ref_frame=$2
module load gcc/8.4.0 python/3.7.7
source /home/lobato/venvs/ofco/bin/activate
# source /home/aymanns/ofco/examples/venvs/venv-for-registration/bin/activate
echo STARTING AT `date`
# python /home/aymanns/ofco/examples/register.py ${folder} ${ref_frame}
python /home/lobato/registration/warping_cluster.py ${folder} ${ref_frame}
echo FINISHED at `date`
