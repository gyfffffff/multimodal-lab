from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
class dataset(Dataset):
    def __init__(self, phase):
        if phase == 'train':
            self.index_label = "data/train.txt"
        elif phase == 'val':
            self.index_label = "data/val.txt"
        elif phase == 'test':
            self.index_label = "data/test_without_label.txt"
        with open(self.index_label) as f:
            f.readline()
            self.index_label = f.readlines()

    def __getitem__(self, index):
        data_index, label = self.index_label[index].split(",")[0], self.index_label[index].split(",")[1]
        img = Image.open("data/data/" + data_index + '.jpg')
        text = open("data/data/" + data_index + '.txt').readlines()[0]
        # img 转tensor
        transf = transforms.Compose([transforms.ToTensor()])
        img = transf(img)

        return (img, text, label)

    def __len__(self):
        return len(self.index_label)
    
def batch_process(batch):
    # batch: list of (img, text)
    # img: tensor
    # text: str
    transformer = torch.nn.Sequential(
        transforms.Resize((256, 256)),  # 防止有些图片太小
        transforms.RandomCrop(224),
    )
    imgs = [transformer(img) for img, text, label in batch]
    imgs_tensor = torch.stack(imgs)
    texts = [text for img, text, label in batch]
    labels = set([label for img, text, label in batch])
    label2idx = {label:idx for idx, label in enumerate(labels)}
    labels = [label2idx[label] for img, text, label in batch]
    labels = torch.tensor(labels)
    
    return imgs_tensor, texts, labels


# def wash(texts):
#     # 去除停用词
#     stopwords = open('data/stopwords.txt').readlines()

# def bert_for_train():
#     path = 'data/train.txt'
#     with open(path, 'r') as f:
#         f.readline()
#         lines = f.readlines()
#     idxs = [line.split(',')[0] for line in lines]
#     texts = [open('data/data/'+idx+'.txt').readlines()[0] for idx in idxs]
#     texts = wash(texts)
#     from sklearn.feature_extraction.text import bertVectorizer
#     bertvectorizer = bertVectorizer()
#     text_feature = bertvectorizer.fit_transform(texts).toarray()
#     idx2bert = zip(idxs, text_feature)
#     print(text_feature.shape)
#     import pickle
#     with open('data/bertvectorizer.pkl', 'wb') as f:
#         pickle.dump(bertvectorizer, f)
#     return 
# if __name__ == "__main__":
#     wash(['a', 'b', 'c'])



