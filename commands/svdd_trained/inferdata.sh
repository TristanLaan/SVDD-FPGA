#!/bin/bash
bsub -J rundim5 -q short7   python tools/svdd-default.py  --dim 5  --fixed_target 1  --run True --modeldir models_trained --maxEvents 1000
bsub -J rundim8 -q short7   python tools/svdd-default.py  --dim 8  --fixed_target 1    --run True --modeldir models_trained 
bsub -J rundim13 -q short7  python tools/svdd-default.py  --dim 13  --fixed_target 1    --run True --modeldir models_trained 
bsub -J rundim21 -q short7  python tools/svdd-default.py  --dim 21  --fixed_target 1    --run True --modeldir models_trained 
bsub -J rundim32 -q short7  python tools/svdd-default.py  --dim 34  --fixed_target 1    --run True --modeldir models_trained 
bsub -J rundim55 -q short7  python tools/svdd-default.py  --dim 55  --fixed_target 1    --run True --modeldir models_trained 
bsub -J rundim89 -q short7  python tools/svdd-default.py  --dim 89  --fixed_target 1    --run True --modeldir models_trained 
bsub -J rundim144 -q short7 python tools/svdd-default.py  --dim 144  --fixed_target 1    --run True --modeldir models_trained 
bsub -J rundim233 -q short7 python tools/svdd-default.py  --dim 233  --fixed_target 1    --run True --modeldir models_trained 