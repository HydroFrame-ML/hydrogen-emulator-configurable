#!/bin/bash
#
#SBATCH --job-name=resnet_train_3
#SBATCH --output=log_resnet_train_3.txt
#
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=200GB

TODAY=`date +"%Y-%m-%d"`
VARIANT="2l_256hd_revisions"
CONFIG=$(cat <<- EOM
{
    "resume_from_checkpoint": true,
    "train_dataset_files": [
        "/scratch/network/ab6361/pfclm_2003_bitrounded.zarr",
        "/scratch/network/ab6361/pfclm_2004_bitrounded.zarr",
        "/scratch/network/ab6361/pfclm_2005_bitrounded.zarr"
    ],
    "scaler_file": "/home/ab6361/hydrogen_workspace/data/new_scalers_may8.scalers",
    "log_dir": "/home/ab6361/hydrogen_workspace/artifacts/revisions_logs",
    "run_name": "resnet_$VARIANT",
    "forcings": ["APCP", "Temp_max", "Temp_min", "melt", "et"],
    "parameters": [
        "topographic_index",
        "elevation",
        "frac_dist",
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
    "learning_rate": 0.001,
    "gradient_loss_penalty": true,
    "sequence_length": 120,
    "patch_size": 64,
    "batch_size": 8,
    "num_workers": 8,
    "max_epochs": 9,
    "precision": "bf16",
    "logging_frequency": 10,
    "model_def": {
        "type": "MultiStepModel",
        "config": {
            "layer_model": "BasicResNet",
            "in_channel":25,
            "out_channel": 5,
            "layer_model_kwargs": {
                "hidden_dim": 256,
                "depth": 2
            }
        }
    }
}
EOM
)
echo $CONFIG > config_resnet_phase_3.json
parflow_emulator --mode train --domain subsurface --config config_resnet_phase_3.json