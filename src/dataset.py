from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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
        data_index = self.index_label[index].split(",")[0]
        img = Image.open("data/data/" + data_index + '.jpg')
        text = open("data/data/" + data_index + '.txt').readlines()[0]
        # img 转tensor
        transf = transforms.Compose([transforms.ToTensor()])
        img = transf(img)

        return (img, text)

    def __len__(self):
        return len(self.index_label)
    
def batch_process(batch):
    wlist, hlist = [], []
    for img, text in batch:
        imgSize = img.shape
        # print(33, imgSize)
        wlist.append(imgSize[1])
        hlist.append(imgSize[2])
    maxw = max(wlist)
    maxh = max(hlist)
    for i in range(len(batch)):
        img, text = batch[i]
        img = transforms.Resize((maxw, maxh))(img)
        batch[i] = (img, text)
    return batch


def wash(texts):
    # 去除停用词
    stopwords = open('data/stopwords.txt').readlines()

def bert_for_train():
    path = 'data/train.txt'
    with open(path, 'r') as f:
        f.readline()
        lines = f.readlines()
    idxs = [line.split(',')[0] for line in lines]
    texts = [open('data/data/'+idx+'.txt').readlines()[0] for idx in idxs]
    texts = wash(texts)
    from sklearn.feature_extraction.text import bertVectorizer
    bertvectorizer = bertVectorizer()
    text_feature = bertvectorizer.fit_transform(texts).toarray()
    idx2bert = zip(idxs, text_feature)
    print(text_feature.shape)
    import pickle
    with open('data/bertvectorizer.pkl', 'wb') as f:
        pickle.dump(bertvectorizer, f)
    return 
if __name__ == "__main__":
    wash(['a', 'b', 'c'])



