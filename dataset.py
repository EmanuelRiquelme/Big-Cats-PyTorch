import torch
from torch.utils.data import Dataset
import os
from  PIL import Image
from torchvision import transforms

class Big_Cats(Dataset):
    def __init__(self, root_dir = 'animals',transform = False):
        self.root_dir = root_dir
        self.labels = self.__getlabels__()
        self.transform = transform if transform else transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((192,192)),
                transforms.RandomHorizontalFlip(p=.3),
                transforms.Normalize(mean = (.5,.5,.5),std = (.5,.5,.5)),
            ])


    def __getlabels__(self):
        files = [file for file in os.listdir(self.root_dir)]
        labels = list(set(files))
        return dict(zip(labels,torch.arange(len(labels))))

    def __getfiles__(self):
        files = []
        for sub_dir in os.listdir(self.root_dir):
            files.extend([f'{self.root_dir}/{sub_dir}/{file}' for file in os.listdir(f'{self.root_dir}/{sub_dir}')])
        return files

    def __len__(self):
        return len(self.__getfiles__())

    def __getitem__(self,idx):
        file_name = self.__getfiles__()[idx]
        img = self.transform(Image.open(file_name))
        label =  self.labels[file_name.split('/')[1]]
        return img,label
