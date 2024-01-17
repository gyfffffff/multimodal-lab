import torch
from torch import nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel


class BertResnet(nn.Module):
    def __init__(self, args):
        super(BertResnet, self).__init__()
        self.args = args
        if self.args.use_image:
            self.resnet = models.resnet18(pretrained=False)
            state_dict = torch.load('src/model/resnet18-5c106cde.pth')
            self.resnet.load_state_dict(state_dict)  # 512维
            self.resnet.fc = nn.Identity()
        if self.args.use_text:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(512+768, 3)
        self.sigmoid = nn.Sigmoid() 
        
        
    def forward(self, batch):
        imgs, texts = batch  # imgs: [batchsize, 3, 224, 224], texts: list of str, labels: tensor[batchsize]
        if self.args.use_image:
            img_feature = self.resnet(imgs)  # [batchsize, 512]
        if self.args.use_text:
            text_feature_list = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt')
                outputs = self.bert(**inputs)
                text_feature_list.append(outputs.last_hidden_state[:, 0, :])
            text_features = torch.stack(text_feature_list).squeeze()
        feature = torch.cat([img_feature, text_features], dim=1)

        output = self.fc(feature)
        output = self.sigmoid(output)
        return output

