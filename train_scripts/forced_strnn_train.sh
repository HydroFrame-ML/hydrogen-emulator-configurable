#!/bin/bash
#
#SBATCH --job-name=fstrnn_train
#SBATCH --output=fstrnn_train.txt
#
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=26:00:00
#SBATCH --mem=200GB

# source /home/ab6361/.bashrc
# source activate all

TODAY=`date +"%Y-%m-%d"`
VARIANT="workflow_test"
CONFIG=$(cat <<- EOM
{
    "train_dataset_files": [
        "/scratch/ab6361/pfclm_conus1_zarr/conus1_2003_preprocessed.zarr",
        "/scratch/ab6361/pfclm_conus1_zarr/conus1_2004_preprocessed_fix.zarr"
    ],
    "valid_dataset_files": [
        "/scratch/ab6361/pfclm_conus1_zarr/conus1_2005_preprocessed.zarr"
    ],
    "scaler_file": "/home/ab6361/hydrogen_workspace/model_staging/unet_configurable/conus1.scalers",
    "resume_from_checkpoint": false,
    "log_dir": "/home/ab6361/hydrogen_workspace/artifacts/configurable_logs",
    "run_name": "forced_strnn_train_$VARIANT",
    "forcings": ["APCP", "melt", "et"],
    "parameters": [
        "topographic_index",
        "permeability_0",
        "permeability_1",
        "permeability_2",
        "permeability_3",
        "permeability_4",
        "porosity_0",
        "porosity_1",
        "porosity_2",
        "porosity_3",
        "porosity_4",
        "van_genuchten_alpha_0",
        "van_genuchten_alpha_1",
        "van_genuchten_alpha_2",
        "van_genuchten_alpha_3",
        "van_genuchten_alpha_4",
        "van_genuchten_n_0",
        "van_genuchten_n_1",
        "van_genuchten_n_2",
        "van_genuchten_n_3",
        "van_genuchten_n_4"
    ],
    "states": [
        "pressure_0",
        "pressure_1",
        "pressure_2",
        "pressure_3",
        "pressure_4"
    ],
    "targets": [
        "pressure_0",
        "pressure_1",
        "pressure_2",
        "pressure_3",
        "pressure_4"
    ],
    "sequence_length": 28,
    "patch_size": 128,
    "batch_size": 6,
    "num_workers": 8,
    "max_epochs": 3,
    "logging_frequency": 100,
    "model_def": {
        "type": "ForcedSTRNN",
        "config": {
            "num_layers": 3,
            "num_hidden": [64, 64, 64],
            "img_channel": 5,
            "out_channel": 5,
            "act_channel": 3,
            "init_cond_channel": 5,
            "static_channel": 21
        }
    }
}
EOM
)
echo $CONFIG > train_config.json
run_emulator --mode train --domain subsurface --config train_config.json
