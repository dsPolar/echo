import torch
from torch.utils import data
import numpy as np
import pickle

#Chroma, Spectral Contrast and Tonnetz in both
#LM in LMC
#MFCC in MC

class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'LMC':
            lm = self.dataset[index]['features']['logmelspec']
            chroma = self.dataset[index]['features']['chroma']
            speccon = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            feature = np.concatenate((lm,chroma,speccon,tonnetz), axis=0)

            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            mc = self.dataset[index]['features']['mfcc']
            chroma = self.dataset[index]['features']['chroma']
            speccon = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            feature = np.concatenate((mc,chroma,speccon,tonnetz), axis=0)
            # Edit here to load and concatenate the neccessary features to
            # create the MC feature
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            lm = self.dataset[index]['features']['logmelspec']
            mc = self.dataset[index]['features']['mfcc']
            chroma = self.dataset[index]['features']['chroma']
            speccon = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            feature = np.concatenate((lm,mc,chroma,speccon,tonnetz), axis=0)
            # Edit here to load and concatenate the neccessary features to
            # create the MLMC feature
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
