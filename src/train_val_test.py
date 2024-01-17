from torch import nn
from logger import logger
from torch.utils.data import DataLoader
from dataset import dataset, batch_process
import torch


class train_val_test:
    def __init__(self, args):
        self.args = args
        self.train_loader = DataLoader(dataset('train'), batch_size=args.batch_size, shuffle=True, collate_fn=batch_process)
        self.val_loader = DataLoader(dataset('val'), batch_size=args.batch_size, shuffle=True, collate_fn=batch_process)
        self.test_loader = DataLoader(dataset('test'), batch_size=args.batch_size, shuffle=False, collate_fn=batch_process)
        self.loss = nn.CrossEntropyLoss()
        self.logger = logger(args.log_dir, args.version)
        self.train_acc_history = []
        self.val_acc_history = []
        self.best_acc = -1
        self.patience = self.args.patience
    def train(self, model):
        model.to(self.args.device)
        self.logger.write_config(self.args)
        for epoch in range(self.args.epochs):
            self.logger.write('Epoch: {}'.format(epoch+1))
            self.train_epoch(model)
            self.val(model)
        self.plot()

    def train_epoch(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        model.train()
        acc = 0
        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.args.device)
            targets = batch[2]
            output = model(batch[:2])  # output: [batchsize, 3]
            loss = self.loss(output, targets)  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc += self.getacc(output, targets)/len(self.train_loader)
            if i % 50 == 0:
                self.logger.write('    batch: {}, loss: {}'.format(i, loss.item()))
        self.logger.write('    train acc: {}'.format(acc))
        self.train_acc_history.append(acc)
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(model.state_dict(), f'res/{self.args.version}.pth')
            self.logger.write('    model saved: res/{}.pth'.format(self.args.version))
            self.patience = self.args.patience
        else:
            self.patience -= 1
        if self.patience == 0:
            self.logger.write('    early stop.')
            raise KeyboardInterrupt
    
    def val(self, model):
        model.eval()
        acc = 0
        for batch in self.val_loader:
            batch = batch.to(self.args.device)
            targets = batch[2]
            output = model(batch[:2])
            acc += self.getacc(output, targets)/len(self.val_loader)
        self.logger.write('    val acc: {}'.format(acc))
        self.val_acc_history.append(acc)

    def test(self, model):
        model.to(self.args.device)
        model.eval()
        outputs = []
        for batch in self.test_loader:
            output = model(batch[:2])
            outputs.extend(output)
        self.saveres(outputs)
        self.logger.write('tested')

    def getacc(self, output, targets):
        pred = output.argmax(dim=1)
        acc = (pred == targets).sum().item() / len(pred)
        return acc
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.title('Accuracy')
        plt.plot(self.train_acc_history, label='train')
        plt.plot(self.val_acc_history, label='val')
        plt.legend()
        plt.savefig(f'res/{self.args.version}.png')
        self.logger.write(f'plot saved: res/{self.args.version}.png')

    def saveres(self, outputs):
        import pandas as pd
        pred = outputs.argmax(dim=1)
        pd = pd.read_csv('data/test_without_label.txt')
        pd['tag'] = pred
        pd.to_csv(f'result.csv', index=False)
        self.logger.write(f'res saved.')