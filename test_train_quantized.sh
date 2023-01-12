python tools/train-svdd.py  --dim 5  --fixed_target 1 --hidden_layers "512 256 128" --run True --train True --modeldir models_quantised --quantised True
python tools/train-svdd.py  --dim 55  --fixed_target 1 --hidden_layers "512 256 128" --run True --modeldir models_quantised  --quantised True
python tools/train-svdd.py  --dim 233  --fixed_target 1 --hidden_layers "512 256 128" --run True --modeldir models_quantised  --quantised True




# python tools/svdd-default.py  --dim 55  --fixed_target 1 --hidden_layers "512 256 128" --run True
# python tools/svdd-default.py  --dim 233 --fixed_target 1 --hidden_layers "512 256 128" --run True


# --iterations $4 --batch $5 --device "cpu" --precision $7 
# --dim $1 --hidden_layers "$2" --fixed_target $3 --iterations $4 --batch $5 --device "cpu" --precision $7 