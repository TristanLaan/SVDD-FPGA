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

python tools/plot.py --plotdir plots_quantised_models \
 --modeldir models_quantised \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_8 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_13 \
 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_21 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_34 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_89 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_144 \
#  --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233 \

