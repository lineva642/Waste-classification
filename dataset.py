import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torchvision import transforms as T
import cv2
import torch
from torch.utils.data.dataset import Dataset

#augmentation
def get_train_transform(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter (brightness=0.07, contrast=0.07,
                           saturation=0.1, hue=0.1, always_apply=False, p=0.3)
    ])


def get_test_transform(size = 224):
    return A.Compose([
        A.Resize(size, size)
    ])

#normalization and convertion to tensor
to_tensor_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

class TrainDataset(Dataset):
    def __init__(self, train, transform=None, is_test= False ):
        self.X = train['path']
        self.is_test = is_test
        self.transform = transform
        if not self.is_test:
            self.y = torch.tensor(train.loc[:, ['alucan', 'glass', 'hdpe', 'pet']].values).long()
    
    def __len__(self):
        return len(self.X) 
    
    def __getitem__(self, index):
        image = cv2.imread(self.X[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.is_test:
             return   to_tensor_transform(image)
        else: 
            label = self.y[index]
            return  to_tensor_transform(image) , label