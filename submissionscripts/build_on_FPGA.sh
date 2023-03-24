#!/bin/bash 
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -p xilinx
#SBATCH --tasks-per-node 15

#Finally, some applications run more efficiently if you leave one core free (i.e. specify 15 tasks for a 16-core node); the only way to know this, is to test your applications using both settings.
timestamp() {
  date +"%T" # current time
}

#setup env
timestamp
cd $HOME/workplace/FPGA/SVDD-FPGA
source setup/setupcn7.sh
timestamp
rm -r building_project_logs
mkdir building_project_logs
timestamp
SCRIPT=tools/svdd-hls4ml.py
python3 $SCRIPT  --ap_fixed_width 20 --ap_fixed_int 10  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_05_hls4mlwrapper  --hls4ml True --maxEvents 500 --build True > building_project_logs/log_build.txt 2> building_project_logs/log_error_build.txt
timestamp