#!/bin/bash
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.1"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.2"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.3"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.4"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.5"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.6"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.7"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.8"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest  --epochs 50 --sparsity "0.9"

