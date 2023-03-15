#!/bin/bash
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_005  --epochs 50 --sparsity "0.05"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_01  --epochs 50 --sparsity "0.1"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_02  --epochs 50 --sparsity "0.2"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_03  --epochs 50 --sparsity "0.3"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_04  --epochs 50 --sparsity "0.4"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_05  --epochs 50 --sparsity "0.5"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_06  --epochs 50 --sparsity "0.6"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_07  --epochs 50 --sparsity "0.7"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_08  --epochs 50 --sparsity "0.8"
bsub -J tr_dim5 -q short7   python tools/train-svdd.py  --dim 5  --fixed_target 1  --train True --modeldir models_trained_pruningtest_09  --epochs 50 --sparsity "0.9"

