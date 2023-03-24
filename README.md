# SVDD-FPGA
Training and testing SVDD's for deployment on FPGA



more info at:
https://www.nikhef.nl/pdp/computing-course/software/where-to-get.html#creating-a-virtualenv-with-conda


#
Setup the Conda enviroment for the first time
#
'''
conda hls4ml-FPGA create --prefix ./env --file setup/environment.yml
conda activate hls4ml-FPGA
'''

#
Setup the Conda enviroment later
#
'''
source setupenv.sh
'''

#
install hls4ml vitis_port branch using
#
'''
python3 setup.py install --root /home/bryan_esc/.local/bin
'''