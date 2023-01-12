#!/bin/bash

python tools/plot.py --modeldir models_conventional --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233
python tools/plot.py --modeldir models_trained --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233


python tools/plot.py --plotdir models_quantised --modeldir models_quantised --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_5 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_55 --model SVDD_3l_512_256_128_bs_10000_ordered_ft_1_zdim_233
