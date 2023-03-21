from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import random
# import struct
from PIL import Image
import numpy as np

class cifarDataset(Dataset):
    def __init__(self,train=True,task_loc=None):
        super(cifarDataset,self).__init__()
        self.train=train
        self.dataset = datasets.cifar.CIFAR100(root='pytorch-cifar100/data/', train=train, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,std=np.array([63.0, 62.1, 66.7]) / 255.0),
                                                                                                          ]), download=False)
        targets = np.array(self.dataset.targets)
        self.idxs=[]
        for i in range(100):
            idx=np.where(targets==i)
            self.idxs.append(idx)
        self.task_list = []
        with open(os.path.join(task_loc), 'r') as f:
            for line in f.readlines():
                curLine = line.strip().split()
                task = list(map(int, curLine))
                self.task_list.append(task)
    def __getitem__(self, idx):
        task=random.choice(self.task_list)
        prompt=torch.zeros(100).scatter_(0,torch.tensor(task),1)
        task = [int(i) for i in task]
        embed=torch.LongTensor(task)

        label=task.index(random.choice(task))

        image_list=self.idxs[label][0].tolist()

        image_idx=random.choice(image_list)

        image=self.dataset[int(image_idx)][0]

        return prompt,image,int(label),embed

    def __len__(self):

        if self.train:
            return 500000
        else:
            return 10000


