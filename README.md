# numerai_NN_example
Numerai tournament example scripts using pytorch NN, lightGBM and optuna 
> https://numer.ai/tournament

Performance of my model based on this example
> [numerai model page](https://numer.ai/emerald_)  
> [numerai model page](https://numer.ai/sapphire_)
# Updates
***
* 2022/04/25  Faster MLP model (nn.ModuleList -> nn.Sequential)  &  adopt numerai V4 data
***

# Features
***
* era-boosted train, time-series cross-validation  
* era-batches training  
* model hyperparameter tuning on pytorch NN and GBDT model  
* several tips on Numerai Forum are also included
***
# Prerequisites
```
python3
gpu environment for pytorch # if you use pytorch NN model
virtualenv
```

# Get this code and build environment
```
git clone https://github.com/meaten/numerai_NN_example.git
cd numerai_NN_example
mkdir env
virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
```

# Quick demo
train model by era-boosted training. you can choose other config files also.
```
python src/main.py --config_file config/mlp.yml --gpu GPU_ID
```

test model for diagnostic.
```
python src/main.py --config_file config/mlp.yml --mode test --gpu GPU_ID
```
inference & submit.  
Please specify follows.  
* pairs of your model name and config file in src/main.py.
* Numerai API user id and secret key in src/default_param.py
```
python src/main.py --mode submit --gpu GPU_ID
```
tune model hyperparameter by optuna
To train with tuned parameters, add ```LOAD_TUNED: True``` to the config file.
```
python src/main.py --config_file config/mlp.yml --mode tune --gpu GPU_ID
```
# LICENCE
MIT

# FEEDBACK
Please send me the bug report or wanted features on GitHub Issue or Numerai Forum.

# SUPPORT
If you find this repository helpful and feel generous, Please send NMR to my wallet address below.
> 0x0000000000000000000000000000000000025769

