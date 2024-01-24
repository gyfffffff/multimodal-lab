from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from transformers import BertTokenizer
from tqdm import tqdm

class dataset(Dataset):
    def __init__(self, phase):
        if phase == 'train':
            self.index_label = "data/train.txt"
        elif phase == 'val':
            self.index_label = "data/val.txt"
        elif phase == 'test':
            self.index_label = "data/test_without_label.txt"
        elif phase == 'train_small':
            self.index_label = "data/train_small.txt"
        elif phase == 'val_small':
            self.index_label = "data/val_small.txt"
        with open(self.index_label) as f:
            f.readline()
            self.index_label = f.readlines()
        self.index_label = [i.strip().split(',') for i in self.index_label]  # [[index, label], ...
        self.text_files = [i[0] + '.txt' for i in self.index_label]  # [index.txt, ...]
        self.img_files = [i[0] + '.jpg' for i in self.index_label]   # [index.jpg, ...]
        label2idx = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}
        self.labels = [label2idx[i[1]] for i in self.index_label]    # [0, 1, 2, ...]
        self.labels = torch.tensor(self.labels)

        self.text_ids_list, self.attention_mask_list = [], []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for f in tqdm(self.text_files, desc=f"{phase} text tokenize"):
            with open("data/data/" + f, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().encode("ascii", "ignore").decode()
                output = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=150)
                text_ids = output['input_ids'].squeeze()
                attention_mask = output['attention_mask'].squeeze()
                self.text_ids_list.append(text_ids)
                self.attention_mask_list.append(attention_mask)
        
        self.img_list = []
        for f in tqdm(self.img_files, desc=f"{phase} img transform"):
            img = Image.open("data/data/" + f)
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 防止有些图片太小
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
            img = transform(img)
            self.img_list.append(img)
                
        
    def __getitem__(self, index):
        data = (self.text_ids_list[index], self.attention_mask_list[index], self.img_list[index], self.labels[index])
        return data

    def __len__(self):
        return len(self.index_label)
    
def trainloader(args):
    if args.train_small:
        trainset = dataset('train_small')
    else:
        trainset = dataset('train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    return trainloader

def valloader(args):
    if args.train_small:
        valset = dataset('val_small')
    else:
        valset = dataset('val')
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    return valloader

def testloader(args):
    testset = dataset('test')
    testloader = DataLoader(testset, batch_size=4, shuffle=False)
    return testloader





