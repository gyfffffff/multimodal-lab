from torch.utils.data import DataLoader
from torch import nn
import torch

class tfidf(nn.Module):
    def __init__(self, args):
        pass
    def forward(self, X):
        pass    

class bert(nn.Module):
    def __init__(self, args):
        pass
    def forward(self, X):
        pass

class tfidf_bert(nn.Module):
    def __init__(self, args):
        super(tfidf_bert, self).__init__()
        self.tfidf = tfidf(args)
        self.bert = bert(args)
    def forward(self, X):
        X_text = X[0]
        X_img = X[1]
        X_text = self.tfidf(X_text)
        X_img = self.bert(X_img)
        X = torch.cat((X_text, X_img), dim=1)

        