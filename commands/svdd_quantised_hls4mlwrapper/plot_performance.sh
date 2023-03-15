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



# python tools/plot.py --plotdir plots_models_quantised_hls4mlwrapper \
#  --modeldir models_quantised_hls4mlwrapper \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_8 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_13 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_21 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_34 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_89 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233 \

# python tools/plot.py --plotdir models_trained_32_6 --AUC True \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_8 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_13 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_21 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_34 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_89 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233 \

# python tools/plot.py --plotdir models_trained_32_6 --AUC True \
#  --referenceAUC "1.0"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_8 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_13 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_21 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_34 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_89 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --labels "<32,6>"\
#  --modeldir models_trained_32_6 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233 \
#  --labels "<32,6>"\


python tools/plot.py --plotdir models_quantised_hls4mlwrapper_intvar_ft_1_zdim144 --AUC True \
 --refmodeldir models_trained_32_6\
 --modeldir models_quantised_hls4mlwrapper_24_2 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<24,2>"\
 --modeldir models_quantised_hls4mlwrapper_26_4 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<26,4>"\
 --modeldir models_quantised_hls4mlwrapper_29_7 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<29,7>"\
 --modeldir models_quantised_hls4mlwrapper_31_9 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<31,9>"\
 --modeldir models_quantised_hls4mlwrapper_34_12 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<34,12>"\
 --modeldir models_quantised_hls4mlwrapper_36_14 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<36,14>"\
 --modeldir models_quantised_hls4mlwrapper_39_17 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<39,17>"\
 --modeldir models_quantised_hls4mlwrapper_41_19 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<41,19>"\
 --modeldir models_quantised_hls4mlwrapper_44_22 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<44,22>"\
 --modeldir models_quantised_hls4mlwrapper_46_24 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<46,24>"\
 --modeldir models_quantised_hls4mlwrapper_48_27 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<48,27>"\
 --modeldir models_quantised_hls4mlwrapper_51_29 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<51,29>"\
 --modeldir models_quantised_hls4mlwrapper_54_32 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<54,32>"\
 --modeldir models_quantised_hls4mlwrapper_57_35 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<57,35>"\
 --modeldir models_quantised_hls4mlwrapper_59_37 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<59,37>"\
 --modeldir models_quantised_hls4mlwrapper_62_40 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<62,40>"\
 --modeldir models_quantised_hls4mlwrapper_64_42 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<64,42>"

python tools/plot.py --plotdir models_quantised_hls4mlwrapper_widthvar_ft_1_zdim144 --AUC True \
 --refmodeldir models_trained_32_6\
 --modeldir models_quantised_hls4mlwrapper_12_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<12,10>"\
 --modeldir models_quantised_hls4mlwrapper_15_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<15,10>"\
 --modeldir models_quantised_hls4mlwrapper_17_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<17,10>"\
 --modeldir models_quantised_hls4mlwrapper_20_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<20,10>"\
 --modeldir models_quantised_hls4mlwrapper_22_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<22,10>"\
 --modeldir models_quantised_hls4mlwrapper_25_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<25,10>"\
 --modeldir models_quantised_hls4mlwrapper_27_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<27,10>"\
 --modeldir models_quantised_hls4mlwrapper_30_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<30,10>"\
 --modeldir models_quantised_hls4mlwrapper_32_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<32,10>"\
 --modeldir models_quantised_hls4mlwrapper_35_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<35,10>"\
 --modeldir models_quantised_hls4mlwrapper_37_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<37,10>"\
 --modeldir models_quantised_hls4mlwrapper_40_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<40,10>"\
 --modeldir models_quantised_hls4mlwrapper_43_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<43,10>"\
 --modeldir models_quantised_hls4mlwrapper_46_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<46,10>"\
 --modeldir models_quantised_hls4mlwrapper_49_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<49,10>"\
 --modeldir models_quantised_hls4mlwrapper_52_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<52,10>"\
 --modeldir models_quantised_hls4mlwrapper_56_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<56,10>"\
 --modeldir models_quantised_hls4mlwrapper_59_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<59,10>"\
 --modeldir models_quantised_hls4mlwrapper_64_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
 --labels "<64,10>"

python tools/plot.py --plotdir models_quantised_hls4mlwrapper_intvar_ft_1_zdim5 --AUC True \
 --refmodeldir models_trained_32_6\
 --modeldir models_quantised_hls4mlwrapper_24_2 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<24,2>"\
 --modeldir models_quantised_hls4mlwrapper_26_4 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<26,4>"\
 --modeldir models_quantised_hls4mlwrapper_29_7 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<29,7>"\
 --modeldir models_quantised_hls4mlwrapper_31_9 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<31,9>"\
 --modeldir models_quantised_hls4mlwrapper_34_12 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<34,12>"\
 --modeldir models_quantised_hls4mlwrapper_36_14 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<36,14>"\
 --modeldir models_quantised_hls4mlwrapper_39_17 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<39,17>"\
 --modeldir models_quantised_hls4mlwrapper_41_19 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<41,19>"\
 --modeldir models_quantised_hls4mlwrapper_44_22 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<44,22>"\
 --modeldir models_quantised_hls4mlwrapper_46_24 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<46,24>"\
 --modeldir models_quantised_hls4mlwrapper_48_27 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<48,27>"\
 --modeldir models_quantised_hls4mlwrapper_51_29 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<51,29>"\
 --modeldir models_quantised_hls4mlwrapper_54_32 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<54,32>"\
 --modeldir models_quantised_hls4mlwrapper_57_35 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<57,35>"\
 --modeldir models_quantised_hls4mlwrapper_59_37 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<59,37>"\
 --modeldir models_quantised_hls4mlwrapper_62_40 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<62,40>"\
 --modeldir models_quantised_hls4mlwrapper_64_42 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<64,42>"

python tools/plot.py --plotdir models_quantised_hls4mlwrapper_widthvar_ft_1_zdim5 --AUC True \
 --refmodeldir models_trained_32_6\
 --modeldir models_quantised_hls4mlwrapper_12_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<12,10>"\
 --modeldir models_quantised_hls4mlwrapper_15_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<15,10>"\
 --modeldir models_quantised_hls4mlwrapper_17_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<17,10>"\
 --modeldir models_quantised_hls4mlwrapper_20_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<20,10>"\
 --modeldir models_quantised_hls4mlwrapper_22_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<22,10>"\
 --modeldir models_quantised_hls4mlwrapper_25_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<25,10>"\
 --modeldir models_quantised_hls4mlwrapper_27_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<27,10>"\
 --modeldir models_quantised_hls4mlwrapper_30_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<30,10>"\
 --modeldir models_quantised_hls4mlwrapper_32_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<32,10>"\
 --modeldir models_quantised_hls4mlwrapper_35_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<35,10>"\
 --modeldir models_quantised_hls4mlwrapper_37_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<37,10>"\
 --modeldir models_quantised_hls4mlwrapper_40_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<40,10>"\
 --modeldir models_quantised_hls4mlwrapper_43_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<43,10>"\
 --modeldir models_quantised_hls4mlwrapper_46_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<46,10>"\
 --modeldir models_quantised_hls4mlwrapper_49_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<49,10>"\
 --modeldir models_quantised_hls4mlwrapper_52_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<52,10>"\
 --modeldir models_quantised_hls4mlwrapper_56_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<56,10>"\
 --modeldir models_quantised_hls4mlwrapper_59_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<59,10>"\
 --modeldir models_quantised_hls4mlwrapper_64_10 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
 --labels "<64,10>"


#  bsub -J testhls10 -q short7 python tools/svdd-hls4ml.py --ap_fixed_width 10 --ap_fixed_int 2  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls12 -q short7 python tools/svdd-hls4ml.py --ap_fixed_width 12 --ap_fixed_int 4  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls15 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 15 --ap_fixed_int 7   --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls17 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 17 --ap_fixed_int 9   --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls20 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 20 --ap_fixed_int 12   --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls22 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 22 --ap_fixed_int 14  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls25 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 25 --ap_fixed_int 17  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls27 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 27 --ap_fixed_int 19  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls30 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 30 --ap_fixed_int 22  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls32 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 32 --ap_fixed_int 24  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls35 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 35 --ap_fixed_int 27  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls37 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 37 --ap_fixed_int 29  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls40 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 40 --ap_fixed_int 32  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000


# bsub -J testhls12 -q short7 python tools/svdd-hls4ml.py --ap_fixed_width 12 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls15 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 15 --ap_fixed_int 10   --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls17 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 17 --ap_fixed_int 10   --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls20 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 20 --ap_fixed_int 10   --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls22 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 22 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls25 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 25 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls27 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 27 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls30 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 30 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls32 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 32 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls35 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 35 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls37 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 37 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls40 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 40 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls43 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 43 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls46 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 46 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls49 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 49 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls52 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 52 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls56 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 56 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls59 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 59 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000
# bsub -J testhls64 -q short7 python tools/svdd-hls4ml.py  --ap_fixed_width 64 --ap_fixed_int 10  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper  --hls4ml True --maxEvents 50000


