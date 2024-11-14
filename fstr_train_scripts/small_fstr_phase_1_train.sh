#!/bin/bash
#
#SBATCH --job-name=fstr_train_1
#SBATCH --output=log_fstr_train.txt
#
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=128GB

export http_proxy=http://verde:8080
export https_proxy=http://verde:8080
source ~/.mlflow_credentials

TODAY=`date +"%Y-%m-%d"`
CONFIG=$(cat <<- EOM
{
    "resume_from_checkpoint": false,
    "train_dataset_files": [
        "/scratch/network/ab6361/pfclm_2004_bitrounded.zarr",
        "/scratch/network/ab6361/pfclm_2005_bitrounded.zarr"
    ],
    "run_name": "hydrogen_fstr_layers-2_hidden-16",
    "forcings": ["APCP", "Temp_max", "Temp_min", "melt", "et"],
    "parameters": [
        "topographic_index",
        "permeability_0",
        "permeability_1",
        "permeability_2",
        "porosity_0",
        "porosity_1",
        "porosity_2",
        "van_genuchten_alpha_0",
        "van_genuchten_alpha_1",
        "van_genuchten_alpha_2",
        "van_genuchten_n_0",
        "van_genuchten_n_1",
        "van_genuchten_n_2"
    ],
    "states": [
        "pressure_prev_0",
        "pressure_prev_1",
        "pressure_prev_2",
        "pressure_prev_3",
        "pressure_prev_4"
    ],
    "targets": [
        "pressure_0",
        "pressure_1",
        "pressure_2",
        "pressure_3",
        "pressure_4"
    ],
    "learning_rate": 0.01,
    "gradient_loss_penalty": true,
    "sequence_length": 2,
    "patch_size": 256,
    "batch_size": 32,
    "num_workers": 16,
    "max_epochs": 1,
    "logging_frequency": 1,
    "precision": 16,
    "model_type": "ForcedSTRNN",
    "model_config": {
        "num_layers": 2,
        "num_hidden": [16, 16],
        "img_channel": 5,
        "out_channel": 5,
        "act_channel": 5,
        "init_cond_channel": 5,
        "static_channel": 13
    }
}
EOM
)
echo $CONFIG > config_train_fstr.json
python runner.py --mode train --domain subsurface --config config_train_fstr.json
