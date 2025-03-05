import yaml
import torch

from dataset import ParFlowDataset
from model import get_model
from train import train_model
from argparse import ArgumentParser
from utils import get_optimizer, get_loss, get_dtype
from torch.utils.data import DataLoader

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def custom_collate(batch):
    s, e, p, y = [], [], [], []
    for b in batch:
        s.append(b[0])
        e.append(b[1])
        p.append(b[2])
        y.append(b[3])
    s = torch.stack(s)
    e = torch.stack(e)
    p = torch.stack(p)
    y = torch.stack(y)
    return s, e, p, y

def train(
    name: str,
    log_location: str,
    model_type: str,
    optimizer: str,
    loss: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    data_def: dict,
    model_def: dict,
    device: str,
    num_workers: int,
    dtype: str,
    **kwargs
):
    # Create the data loader
    dtype = get_dtype(dtype)
    dataset = ParFlowDataset(**data_def, dtype=dtype)
    train_dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=custom_collate, 
        shuffle=True, 
        num_workers=num_workers
    )


    # Create the model
    # Add names of model inputs to model definition for scaling, if needed
    model_def['pressure_names'] = dataset.PRESSURE_NAMES
    model_def['evaptrans_names'] = dataset.EVAPTRANS_NAMES
    model_def['param_names'] = dataset.PARAM_NAMES
    model_def['n_evaptrans'] = dataset.n_evaptrans
    model_def['parameter_list'] = dataset.parameter_list
    model_def['param_nlayer'] = dataset.param_nlayer
    model = get_model(model_type, model_def)
    model = model.to(device).to(dtype)


    # Create the optimizer and loss function
    optimizer = get_optimizer(optimizer, model, lr)
    loss_fn = get_loss(loss)

    metrics = train_model(
        model, train_dl, optimizer, loss_fn, n_epochs, device=device
    )
    print('----------------------------------------')
    print(metrics)
    print('----------------------------------------')
    
    metrics_filename = f'{log_location}/{name}_metrics.csv'
    weights_filename = f'{log_location}/{name}_weights_only.pth'
    model_filename = f'{log_location}/{name}_model.pth'
    metrics.to_csv(metrics_filename)
    torch.save(model.state_dict(), weights_filename)
    m = torch.jit.script(model)
    torch.jit.save(m, model_filename)

    print('----------------------------------------')
    print(f'Metrics saved to {metrics_filename}')
    print(f'Model saved to {model_filename}')


def test():
    pass

def main(config, mode):
    config = read_config(config)

    if mode == "train":
        print("TRAINING")
        train(**config)
    elif mode == "test":
        print("TESTING")
        # Note: Not implemented
        test(**config)


if __name__ == "__main__":
    # EXAMPLE USAGE: python main.py --config example_config.yaml --mode train
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "test"], default="train"
    )
    args = parser.parse_args()
    main(args.config, args.mode)
