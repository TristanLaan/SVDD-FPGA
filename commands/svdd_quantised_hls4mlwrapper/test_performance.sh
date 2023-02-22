#!/bin/bash
bsub python tools/svdd-hls4ml.py  --dim 5  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 8  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 13  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 21  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 34  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 55  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 89  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 144  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000
bsub python tools/svdd-hls4ml.py  --dim 233  --fixed_target 1  --train True --modeldir models_quantised_hls4mlwrapper --outmodeldir models_hls4mlwrapper --hls4ml True --maxEvents 50000