from torch import nn
import logger
from torch.utils.data import DataLoader
from dataset import dataset, batch_process


class train_val_test:
    def __init__(self, args):
        self.args = args
        self.train_loader = DataLoader(dataset('train'), batch_size=args.batch_size, shuffle=True, collate_fn=batch_process)

    def train(self, model):
        for epoch in range(self.args.epochs):
            self.train_epoch(model)
            self.val(model)
    def train_epoch(self, model):
        for batch in self.train_loader:
            model.train()
            model(batch)
    def val(self, model):
        pass
    def test(self, model):
        pass