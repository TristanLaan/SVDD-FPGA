#!/bin/bash

# python tools/svdd-default.py  --dim 5  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 8  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 13  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 21  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 34  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 55  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 89  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 144  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 233 --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True

python tools/plot.py --plotdir plots_models_trained --make_roc_plots True \
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "ft 1 zdim 5"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_8 \
 --labels "ft 1 zdim 8"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_13 \
 --labels "ft 1 zdim 13"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_21 \
 --labels "ft 1 zdim 21"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_34 \
 --labels "ft 1 zdim 34"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 \
 --labels "ft 1 zdim 55"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_89 \
 --labels "ft 1 zdim 89"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "ft 1 zdim 144"\
 --modeldir models_trained_32_6 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233 \
 --labels "ft 1 zdim 233"





# python tools/plot.py --plotdir models_quantised_hls4mlwrapper_fixedwidth --AUC True \
#  --refmodeldir models_trained_32_6\
#  --modeldir models_quantised_hls4mlwrapper_12_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<12,10>"\
#  --modeldir models_quantised_hls4mlwrapper_15_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<15,10>"\
#  --modeldir models_quantised_hls4mlwrapper_17_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<17,10>"\
#  --modeldir models_quantised_hls4mlwrapper_20_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<20,10>"\
#  --modeldir models_quantised_hls4mlwrapper_22_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<22,10>"\
#  --modeldir models_quantised_hls4mlwrapper_25_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<25,10>"\
#  --modeldir models_quantised_hls4mlwrapper_27_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<27,10>"\
#  --modeldir models_quantised_hls4mlwrapper_30_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<30,10>"\
#  --modeldir models_quantised_hls4mlwrapper_32_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<32,10>"\
#  --modeldir models_quantised_hls4mlwrapper_35_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<35,10>"\
#  --modeldir models_quantised_hls4mlwrapper_37_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<37,10>"\
#  --modeldir models_quantised_hls4mlwrapper_40_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<40,10>"\
#  --modeldir models_quantised_hls4mlwrapper_43_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<43,10>"\
#  --modeldir models_quantised_hls4mlwrapper_46_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<46,10>"\
#  --modeldir models_quantised_hls4mlwrapper_49_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<49,10>"\
#  --modeldir models_quantised_hls4mlwrapper_52_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<52,10>"\
#  --modeldir models_quantised_hls4mlwrapper_56_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<56,10>"\
#  --modeldir models_quantised_hls4mlwrapper_59_10 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<59,10>"\
