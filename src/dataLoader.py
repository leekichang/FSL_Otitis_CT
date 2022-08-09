import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader

class DataManager():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        
    def Load_Dataset(self):
        try:
            X, Y = np.load(f'./{self.dataset_dir}/X.npy'), np.load(f'./{self.dataset_dir}/Y.npy')
            dataset = CT_dataset(X, Y)
            return dataset
        except:
            print('Run pairwise_gen.py first!')
                
    def Load_DataLoader(self, dataset, batch_size, is_train):
        if is_train == True:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False)

class CT_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        self.len = len(self.X)
        return self.len
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx,:]).float()
        Y = torch.tensor(self.Y[idx,:]).float()
        return X, Y
    
if __name__ == '__main__':
    print('dataLoader.py')
    DM = DataManager('./dataset')
    dataset = DM.Load_Dataset()
    dataLoader = DM.Load_DataLoader(dataset, 171, is_train=True)
    for idx, batch in enumerate(dataLoader):
        X, Y = batch
        print(X.shape, Y.shape)