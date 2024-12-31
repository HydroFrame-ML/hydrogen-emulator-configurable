TODAY=`date +"%Y-%m-%d"`
VARIANT="1l_32hd_full"
CONFIG=$(cat <<- EOM
{
    "resume_from_checkpoint": true,
    "train_dataset_files": [
        "/home/andrbenn/data/hydrogen/pfclm_2003_bitrounded.zarr",
        "/home/andrbenn/data/hydrogen/pfclm_2004_bitrounded.zarr",
        "/home/andrbenn/data/hydrogen/pfclm_2005_bitrounded.zarr"
    ],
    "logging_location": "file:/home/andrbenn/data/hydrogen/full_model_runs/",
    "run_name": "fstr_$VARIANT",
    "forcings": ["APCP", "Temp_max", "Temp_min"],
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
        "pressure_prev_4",
        "swe_prev",
        "et_prev"
    ],
    "targets": [
        "pressure_0",
        "pressure_1",
        "pressure_2",
        "pressure_3",
        "pressure_4",
        "swe",
        "et",
        "water_table_depth",
        "streamflow"
    ],
    "learning_rate": 0.001,
    "gradient_loss_penalty": true,
    "sequence_length": 8,
    "patch_size": 64,
    "batch_size": 16,
    "num_workers": 16,
    "max_epochs": 10,
    "logging_frequency": 1,
    "precision": "bf16-mixed",
    "model_type": "ParflowClmEmulator",
    "model_config": {
        "num_layers": 1,
        "num_hidden": [32],
        "img_channel": 7,
        "out_channel": 7,
        "act_channel": 3,
        "init_cond_channel": 7,
        "static_channel": 15, 
        "number_depth_layers": 5
    }
}
EOM
)
echo $CONFIG > config_medium_fstr_phase_2.json
parflow_emulator --mode train --domain subsurface --config config_medium_fstr_phase_2.json


