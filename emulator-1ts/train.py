import torch
import pandas as pd
from tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

def train_epoch(
    model,
    dataset,
    optimizer,
    loss_fn,
    device,
    train=True,
):
    # Trains 1 epoch
    # TODO: Track losses
    prefix = 'train' if train else 'valid'
    for i, batch in enumerate(dataset):
        x, y = batch
        x = x.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        y = y.squeeze()
        if not len(x): continue
        optimizer.zero_grad()
        if train:
            yhat = model(x).squeeze()
        else:
            # Don't compute gradients
            # Saves on computation
            with torch.no_grad():
                yhat = model(x).squeeze()
        if torch.isnan(yhat).any():
            print()
            print(torch.isnan(x).sum(), torch.isnan(yhat).sum())
            print()
            raise ValueError(
                f'Predictions went nan! Nans in input: {torch.isnan(x).sum()}'
            )
        loss = loss_fn(yhat, y)
        if train:
            loss.backward()
            optimizer.step()
        
    return loss

def train_model(
    model, 
    train_dl, 
    opt, 
    loss_fun, 
    max_epochs,
    scheduler=None,
    val_dl=None, 
    device=DEVICE, 
    dtype=DTYPE
):
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for e in (bar := tqdm(range(max_epochs))):
        # Make sure to turn on train mode here
        # so that we update parameters
        model.train()
        train_metrics = train_epoch(
            model, train_dl, opt, loss_fun, train=True, device=device, dtype=dtype
        )
        train_df = train_df._append(train_metrics, ignore_index=True)
        tl = train_metrics['train_loss']

        if val_dl is not None:
            # Now set to evaluation mode which reduces
            # the memory/computational cost
            model.eval()
            valid_metrics = train_epoch(
                model, val_dl, opt, loss_fun, train=False, device=device, dtype=dtype
            )
            valid_df = valid_df._append(valid_metrics, ignore_index=True)
            vl = valid_metrics['valid_loss']

            bar.set_description(f'Train loss: {tl:0.1e}, valid loss: {vl:0.1e}')
        else:
            bar.set_description(
                f'Train loss: {tl:0.1e}'
            )

        # Log our losses and update the status bar
        if scheduler is not None:
            scheduler.step()

    if val_dl is not None:
        # Merge the two dataframes
        train_df = pd.concat([train_df, valid_df], axis=1)
    return train_df