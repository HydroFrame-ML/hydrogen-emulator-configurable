import yaml
import torch

from dataset import ParFlowDataset
from model import get_model
from train import train_model
from argparse import ArgumentParser
from utils import get_optimizer, get_loss
from torch.utils.data import DataLoader

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    **kwargs
):
    # TODO: 1: Check this works
    model = get_model(model_type, model_def)
    model = model.to(device)
    print('----------------------------------------')
    print(model)
    print('To do: Delete if working1')
    print('----------------------------------------')

    # TODO: 1: Check this works
    optimizer = get_optimizer(optimizer, model, lr)
    loss_fn = get_loss(loss)
    print('----------------------------------------')
    print(optimizer)
    print(loss_fn)
    print('To do: Delete if working2')
    print('----------------------------------------')

    
    dataset = ParFlowDataset(**data_def)
    train_dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    print('----------------------------------------')
    print(f'Train dataset has {len(train_dl)} batches')
    x, y = dataset[5]
    #x, y = next(iter(dataset[5])) # This doesn't work... why? 
    print(f'Shape of first batch: {x.shape}, {y.shape}')
    
    print('To do: Delete if working3')
    print('----------------------------------------')
    
    print(f'Train dataloader is working?')
    x, y = next(iter(train_dl))
    
    print('To do: Delete if working4')
    print('----------------------------------------')

    metrics = train_model(
        model, train_dl, optimizer, loss_fn, n_epochs, device=device
    )
    print('----------------------------------------')
    print(metrics)
    print('To do: Delete if working5')
    print('----------------------------------------')
    metrics_filename = f'{log_location}/{name}_metrics.csv'
    model_filename = f'{log_location}/{name}_model.pth'
    metrics.to_csv(metrics_filename)
    torch.save(model.state_dict(), model_filename)
    print('----------------------------------------')
    print(f'Metrics saved to {metrics_filename}')
    print(f'Model saved to {model_filename}')
    print('To do: Delete if working6')


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