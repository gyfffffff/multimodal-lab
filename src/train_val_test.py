from torch import nn
from logger import logger
from dataset import trainloader, valloader, testloader
import torch
from tqdm import tqdm

class train_val_test:
    def __init__(self, args):
        self.args = args
        self.loss = nn.CrossEntropyLoss()
        self.logger = logger(args.log_dir, args.version)
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.best_acc = -1
        self.patience = self.args.patience

    def train(self, model):
        self.train_loader = trainloader(self.args)
        self.val_loader = valloader(self.args)
        model.to(self.args.device)
        self.logger.write_config(self.args)
        for epoch in range(self.args.epochs):
            self.logger.write('Epoch: {}'.format(epoch+1))
            self.train_epoch(model, epoch)
            if not self.args.train_small:
                self.val(model)
            if self.patience == 0:
                self.logger.write('    early stop.')
                break
        self.plot()
        self.test(model)

    def train_epoch(self, model, epoch):
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        if self.args.modelname.lower() != 'roberta_swin_att':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.AdamW([
                {
                    'params': model.text_embedding.roberta.parameters(),
                    'lr': self.args.lr_finetune
                },
                {
                    'params': model.img_embedding.swin.parameters(),
                    'lr': self.args.lr_finetune
                },
                {
                    'params': model.text_embedding.aligner.parameters(),
                    'lr': self.args.lr_downstream
                },
                {
                    'params': model.img_embedding.aligner.parameters(),
                    'lr': self.args.lr_downstream
                },
                {
                    'params': model.fuser.parameters(),
                    'lr': self.args.lr_downstream
                }
            ])
        model.train()
        # total_loss = 0
        acc = 0
        for batch in tqdm(self.train_loader, desc=f'train epoch {epoch+1}'):
            # batch: [tensor[batchsize, 150], tensor[batchsize, 150], tensor[batchsize, 3, 224, 224], tensor[batchsize]]
            text_ids, attention_masks, imgs, targets = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device), batch[3].to(self.args.device)
            output = model(text_ids, attention_masks, imgs)  # output: [batchsize, 3]
            loss = self.loss(output, targets)  
            # total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc += self.getacc(output, targets)/len(self.train_loader)
        self.logger.write('    train loss: {}'.format(loss.item()))
        self.logger.write('    train acc: {}'.format(acc))
        self.train_loss_history.append(loss.item())
        self.train_acc_history.append(acc)

    
    def val(self, model):
        print('    val...')
        model.eval()
        acc = 0
        for batch in self.val_loader:
            text_ids, attention_masks, imgs, targets = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device), batch[3].to(self.args.device)
            output = model(text_ids, attention_masks, imgs)
            acc += self.getacc(output, targets)/len(self.val_loader)
        self.logger.write('    val acc: {}'.format(acc))
        self.val_acc_history.append(acc)
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(model.state_dict(), f'res/{self.args.version}.pth')
            self.logger.write('    model saved: res/{}.pth'.format(self.args.version))
            self.patience = self.args.patience
        else:
            self.patience -= 1

    def test(self, model, state_dict_path):
        self.logger.write('\ntest start')
        self.test_loader = testloader(self.args)
        model.load_state_dict(torch.load(state_dict_path))
        outputs = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(self.test_loader, desc='test'):
                text_ids, attention_masks, imgs = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device)
                output = model(text_ids, attention_masks, imgs)
                outputs.extend(output)
            self.saveres(outputs)

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
        pred = [output.argmax() for output in outputs]
        label2idx = {'positive': 0, 'neutral': 1, 'negative': 2}
        idx2label = {v: k for k, v in label2idx.items()}
        pred = [idx2label[i.item()] for i in pred]
        pd = pd.read_csv('data/test_without_label.txt')
        pd['tag'] = pred
        pd.to_csv(f'submit.txt', index=False)
        self.logger.write(f'res saved.')