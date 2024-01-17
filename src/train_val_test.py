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
        self.test_loader = DataLoader(dataset('test'), batch_size=args.batch_size, shuffle=True, collate_fn=batch_process)
        self.loss = nn.CrossEntropyLoss()
        self.logger = logger(args.log_dir, args.version)
        self.train_acc_history = []
        self.val_acc_history = []
        
    def train(self, model):
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
        for batch in self.train_loader:
            targets = batch[2]
            output = model(batch[:2])  # output: [batchsize, 3]
            loss = self.loss(output, targets)  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc += self.getacc(output, targets)/len(self.train_loader)
        self.logger.write('    train acc: {}'.format(acc))
        self.train_acc_history.append(acc)
    
    def val(self, model):
        model.eval()
        acc = 0
        for batch in self.val_loader:
            targets = batch[2]
            output = model(batch[:2])
            acc += self.getacc(output, targets)/len(self.val_loader)
        self.logger.write('    val acc: {}'.format(acc))
        self.val_acc_history.append(acc)

    def test(self, model):
        model.eval()
        acc = 0
        for batch in self.test_loader:
            targets = batch[2]
            output = model(batch[:2])
            acc += self.getacc(output, targets)/len(self.test_loader)
        self.logger.write('test acc: {}'.format(acc))

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