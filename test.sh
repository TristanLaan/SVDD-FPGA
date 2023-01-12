# python tools/svdd-default.py  --dim 5  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 55  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True
# python tools/svdd-default.py  --dim 233 --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_conventional --run True

python tools/svdd-default.py  --dim 5  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_trained --run True
python tools/svdd-default.py  --dim 55  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_trained --run True
python tools/svdd-default.py  --dim 233 --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_trained --run True


# python tools/svdd-default.py  --dim 5  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_quantised --run True --quantised True
# python tools/svdd-default.py  --dim 55  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_quantised --run True --quantised True
# python tools/svdd-default.py  --dim 233 --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_quantised --run True --quantised True


python tools/svdd-default.py  --dim 5  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_quantised --run True --quantised True
# python tools/svdd-default.py  --dim 55  --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_quantised --run True --quantised True
# python tools/svdd-default.py  --dim 233 --fixed_target 1 --hidden_layers "512 256 128" --modeldir models_quantised --run True --quantised True



# --iterations $4 --batch $5 --device "cpu" --precision $7 
# --dim $1 --hidden_layers "$2" --fixed_target $3 --iterations $4 --batch $5 --device "cpu" --precision $7 