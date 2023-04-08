import torch
import torch.nn as nn
import datetime
import os
from dataset import MnistDataset
from models import MyNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

class Trainer():
    def __init__(self, model, config, device='cuda'):
        print(config['title'])
        hp = config['hyperparam']
        self.device = torch.device(device)
        self.batch_size = int(hp['batch_size'])
        self.lr = float(hp['learning_rate'])
        self.epochs = hp['epochs']
        self.val_every = int(hp['val_every'])

        self.data_path = config['data_path']
        self.model = model.to(device)

        if config['logger'] == 'tensorboard':
            self.curtime= datetime.datetime.now()
            self.writer = SummaryWriter(f"logs/training/{self.curtime.strftime('%m%d%H%M')}")
            print(f"Log saved to: logs/{self.curtime.strftime('%m%d%H%M')}")

        os.mkdir(f"logs/models/{self.curtime.strftime('%m%d%H%M')}")
        self._ld()

    def _ld(self):
        self.train_dataset = DataLoader(MnistDataset(self.data_path, True), self.batch_size, shuffle=True)
        self.val_dataset = DataLoader(MnistDataset(self.data_path, False), self.batch_size, shuffle=True)

    def train(self):
        dummy = torch.randn(1, 1, 28, 28).to(self.device)
        self.writer.add_graph(self.model, dummy)

        ops = torch.optim.Adam(self.model.parameters(), lr= self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            pbar1 = tqdm(self.train_dataset)
            self.model.train()
            for i, sample in enumerate(pbar1):
                img = sample['image'].to(self.device)
                label = sample['label'].to(self.device)
                out = self.model(img).squeeze()
                
                ops.zero_grad()
                loss = criterion(out.float(), label)
                loss.backward()
                ops.step()
                
                pbar1.set_description(
                        f"Epoch: {epoch} | Train | Loss: {loss.item()}"
                    )

                self.writer.add_scalar(f'Loss/loss_{epoch}', loss, global_step=i)
            self.writer.add_scalar(f'Loss/loss', loss, global_step=epoch)

            if epoch%self.val_every == 0:
                pbar2 = tqdm(self.val_dataset)
                self.model.eval()
                for i, sample in enumerate(pbar2):
                    img = sample['image'].to(self.device)
                    label = sample['label'].to(self.device)
                    out = self.model(img).squeeze()
                    
                    loss = criterion(out.float(), label)

                    pbar2.set_description(
                        f"Epoch: {epoch} |  Val  | Loss: {loss.item()}"
                    )
                    
                    self.writer.add_scalars(f'Val', {'target': label[0], 'predicted': out[0].argmax()}, global_step=i)
                torch.save(self.model.state_dict(), f"logs/models/{self.curtime.strftime('%m%d%H%M')}/epoch{epoch}.pth")



def main():
    stream = open('config.yaml', 'r')
    data = yaml.load(stream, Loader=yaml.CLoader)
    model = MyNet()
    trainer = Trainer(model, data)
    trainer.train()

if __name__ == '__main__':
    main()