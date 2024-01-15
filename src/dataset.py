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
    
if __name__ == "__main__":
    train_dataset = dataset('train')
    print(train_dataset[0])