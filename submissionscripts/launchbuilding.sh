#!/bin/bash
mkdir -p joblist



# SVDDname=$1
# zdim=$2
# layers=$3
# ft=$4
# precision=$5

device='xilinx-fpga'
# Jobname="${SVDDname}_${precision}_${device}"
# rm joblist/$Jobname.csv


# echo "jobid,batch size,iterations,id,device" >> joblist/$Jobname.csv

sbatch submissionscripts/build_on_FPGA.sh
echo "submitting job  $device"


echo "check if they are on squeue"
squeue -u bryan_esc
