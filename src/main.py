import yaml
import argparse
from train_val_test import train_val_test
import time

def run(args):
    tvt = train_val_test(args)
    modelname = args.modelname.lower()
    if modelname == 'bert_resnet':
        from model.bert_resnet import bert_resnet
        model = bert_resnet(args)
    tvt.train(model)
    tvt.test(model)
    print('done')


if __name__ == "__main__":
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    parser.add_argument('--version', type=str, default=str(localtime))
    for k, v in config.items():
        parser.add_argument('--'+k, type=type(v), default=v)

    args = parser.parse_args()
    run(args)