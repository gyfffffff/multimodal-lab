import yaml
import argparse
from train_val_test import train_val_test
import time


if __name__ == "__main__":
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    parser.add_argument('--version', type=str, default=str(localtime))
    for k, v in config.items():
        parser.add_argument('--'+k, type=type(v), default=v)
    args = parser.parse_args()
    
    tvt = train_val_test(args)
    modelname = args.modelname.lower()
    if modelname == 'bert_resnet_concat':
        from model.bert_resnet_concat import BertResnet
        model = BertResnet(args)
    elif modelname == 'bert_resnet_weight':
        from model.bert_resnet_weight import BertResnet
        model = BertResnet(args)
    elif modelname == 'roberta_swin_att':
        from model.roberta_swin_att import roberta_swin_att
        model = roberta_swin_att('xlm-roberta-base', 'microsoft/swin-base-patch4-window7-224')
    if args.train:
        tvt.train(model)   # train, val
    else:
        tvt.test(model, f'res/{args.version}.pth')
    