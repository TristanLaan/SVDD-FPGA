#!/bin/bash
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained  --epochs 50
bsub -J tr_dim8 -q short7   python tools/train-svdd.py  --dim 8  --fixed_target 1    --train True --modeldir models_trained   --epochs 50
bsub -J tr_dim13 -q short7  python tools/train-svdd.py  --dim 13  --fixed_target 1    --train True --modeldir models_trained   --epochs 50
bsub -J tr_dim21 -q short7  python tools/train-svdd.py  --dim 21  --fixed_target 1    --train True --modeldir models_trained   --epochs 50
bsub -J tr_dim32 -q short7  python tools/train-svdd.py  --dim 34  --fixed_target 1    --train True --modeldir models_trained   --epochs 50
bsub -J tr_dim55 -q short7  python tools/train-svdd.py  --dim 55  --fixed_target 1    --train True --modeldir models_trained   --epochs 50
bsub -J tr_dim89 -q short7  python tools/train-svdd.py  --dim 89  --fixed_target 1    --train True --modeldir models_trained   --epochs 50
bsub -J tr_dim144 -q short7 python tools/train-svdd.py  --dim 144  --fixed_target 1    --train True --modeldir models_trained   --epochs 50
bsub -J tr_dim233 -q short7 python tools/train-svdd.py  --dim 233  --fixed_target 1    --train True --modeldir models_trained   --epochs 50