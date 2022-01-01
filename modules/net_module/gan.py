import torch
from torch import nn
from torch._C import device
from .lstm_wrapper import LSTM
from tqdm import tqdm
from torch.utils.data import DataLoader

class GAN:

    def __init__(self, window_size, epoch = 10):
        
        self.generator = LSTM()
        self.discriminator = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.Linear(window_size, window_size),
            nn.Sigmoid()
        )

        self.adv_loss = nn.BCELoss()
        self.g_loss = nn.SmoothL1Loss()

        self.G_optim = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.D_optim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.adv_loss = self.adv_loss.cuda()

        self.epoch = epoch

    def train(self, data: DataLoader):
        outputs = []
        tq = tqdm(range(self.epoch), total=self.epoch, desc="Training...")
        for ep in tq:
            for x, y in data:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                # z = torch.randn_like(x, device=x.device)
                fake, _ = self.generator(x)
                self.G_optim.zero_grad()
                self.D_optim.zero_grad()

                g_loss = self.g_loss(fake, y)
                adv_loss = self.adv_loss(self.discriminator(fake.squeeze(dim=-1)), torch.zeros_like(fake.squeeze(dim=-1), device=fake.device))

                real_loss = self.adv_loss(self.discriminator(x[:,:, -1]), torch.ones_like(x[:, :, -1], device=x.device))
                loss = (real_loss + adv_loss) / 2 + 2 * g_loss
                loss.backward()
                self.G_optim.step()
                self.D_optim.step()

                tq.set_postfix({
                    "adv_loss": adv_loss.item(),
                    "D_loss": loss.item()
                })

                if ep == self.epoch - 1:
                    with torch.no_grad():
                        outputs.extend(fake[:, -1].cpu().detach().squeeze(dim=1).numpy().tolist())

        return outputs

    def test(self, data: DataLoader):
        outputs = []
        for x, y in tqdm(data):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out, _ = self.generator(x)
            with torch.no_grad():
                outputs.extend(out[:, -1].cpu().detach().squeeze(dim=1).numpy().tolist())

        return outputs