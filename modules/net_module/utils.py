from typing import Tuple
import torch
from torch import nn
from typing import Tuple, cast
from torch.utils.data import DataLoader
from tqdm import tqdm
EPOCHS = 200

def train_step(net: nn.Module, opt: torch.optim.Optimizer, data_loader: DataLoader, criterion: nn.Module):
    outputs = []
    if torch.cuda.is_available():
        net.cuda()
    tq = tqdm(range(EPOCHS), total=EPOCHS, desc="Training..")
    for ep in tq:
        for x, y in data_loader:
            x, y = cast(torch.Tensor, x), cast(torch.Tensor, y)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out, _ = net(x)

            if ep == EPOCHS - 1:
                with torch.no_grad():
                    outputs.extend(out[:, -1].cpu().detach().squeeze(dim=1).numpy().tolist())

            loss: torch.Tensor = criterion(out, y)
            opt.zero_grad()
            loss.backward()

            opt.step()
            tq.set_postfix({
                "loss": loss.item()
            })
    return outputs


@torch.no_grad()
def test_step(net: nn.Module, data_loader: DataLoader):
    outputs = []
    if torch.cuda.is_available():
        net.cuda()
    for x, y in tqdm(data_loader, total=len(data_loader), desc="Testing..."):
        x, y = cast(torch.Tensor, x), cast(torch.Tensor, y)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        out, _ = net(x)

        with torch.no_grad():
            outputs.extend(out[:, -1].cpu().detach().squeeze(dim=1).numpy().tolist())

    return outputs