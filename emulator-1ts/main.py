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

def custom_collate(batch):
    #print(len(batch))
    #print(batch[0].shape)
    x = []
    y = []
    for b in batch:
        #print(b[0].shape)
        #print(b[1].shape)
        x.append(b[0])
        y.append(b[1])
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y 

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
    model = get_model(model_type, model_def)
    model = model.to(device)  

    optimizer = get_optimizer(optimizer, model, lr)
    loss_fn = get_loss(loss)
    
    dataset = ParFlowDataset(**data_def)
    train_dl = DataLoader(
        dataset, batch_size=batch_size, collate_fn = custom_collate, shuffle=True, num_workers=num_workers
    )
    print('----------------------------------------')
    print(f'Train dataset has {len(train_dl)} batches')
    x, y = next(iter(dataset))
    #x, y = next(iter(dataset[5])) # This doesn't work... why? 
    print(f'Shape of first batch: {x.shape}, {y.shape}')
    
    #x, y = next(iter(train_dl))
    #print(f'Shape of first batch: {x.shape}, {y.shape}')

    metrics = train_model(
        model, train_dl, optimizer, loss_fn, n_epochs, device=device
    )
    print('----------------------------------------')
    print(metrics)
    print('----------------------------------------')
    
    metrics_filename = f'{log_location}/{name}_metrics.csv'
    model_filename = f'{log_location}/{name}_model.pth'
    metrics.to_csv(metrics_filename)
    torch.save(model.state_dict(), model_filename)
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