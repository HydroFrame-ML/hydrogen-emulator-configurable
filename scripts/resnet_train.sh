#!/bin/bash
#
#SBATCH --job-name=resnet
#SBATCH --output=resnet.txt
#
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=128GB

source /home/ab6361/.bashrc
source activate all

TODAY=`date +"%Y-%m-%d"`
VARIANT="configurable_7day_1M"
CONFIG=$(cat <<- EOM
{
    "scaler_file": "/home/ab6361/hydrogen_workspace/model_staging/configurable/conus1.scalers",
    "resume_from_checkpoint": true,
    "log_dir": "/home/ab6361/hydrogen_workspace/artifacts/configurable_logs",
    "run_name": "resnet_train_$VARIANT",
    "forcing_vars": ["APCP", "melt", "et"],
    "surface_parameters": ["topographic_index"],
    "subsurface_parameters": ["porosity", "permeability", "van_genuchten_alpha", "van_genuchten_n"],
    "state_vars": ["pressure"],
    "out_vars": ["pressure_next"],
    "sequence_length": 7,
    "patch_size": 56,
    "patch_stride": 28,
    "batch_size": 32,
    "num_dl_workers": 24,
    "max_epochs": 1,
    "logging_frequency": 100,
    "train_samples_per_epoch": 500000,
    "valid_samples_per_epoch":  500,
    "model_def": {
        "type": "MultiStepMultiLayerModel",
        "model_config": {
            "layer_model": "BasicResNet",
            "probability_of_true_inputs": 0.05,
            "layer_model_kwargs": {
                "hidden_dim": 64,
                "total_depth": 1,
                "res_block_depth": 2
            }
        }
    }
}
EOM
)
echo $CONFIG > resnet_config.json
run_emulator --mode train --domain subsurface --config resnet_config.json
