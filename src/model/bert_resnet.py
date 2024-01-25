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
            self.resnet.load_state_dict(state_dict)  # 1000维
            self.img_fc = nn.Linear(1000, 128)

        if self.args.use_text:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.text_fc = nn.Linear(768, 128)
        
        if self.args.use_image and self.args.use_text:
            self.fusion = nn.Linear(128+128, 3)
        else:
            self.fc = nn.Linear(128, 3)
        self.relu = nn.ReLU()
        
        
    def forward(self, text_ids, attention_masks, imgs):
        if self.args.use_image:
            img_feature = self.resnet(imgs)   # [batchsize, 1000]
            img_feature = self.img_fc(img_feature)    # [batchsize, 128]
            img_feature = self.relu(img_feature)      # [batchsize, 128]
        if self.args.use_text:
            text_feature = self.bert(text_ids, 
                           attention_mask=attention_masks).last_hidden_state[:, 0, :]  # [batchsize, 768]
            text_feature = self.text_fc(text_feature)  # [batchsize, 128]
            text_feature = self.relu(text_feature)     # [batchsize, 128]
        
        if self.args.use_image and self.args.use_text:
            feature = torch.cat([img_feature, text_feature], dim=1)
            output = self.fusion(feature)
        elif self.args.use_image:
            img_feature = self.fc(img_feature)
            output = img_feature
        elif self.args.use_text:
            text_feature = self.fc(text_feature)
            output = text_feature

        return output

