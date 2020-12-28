from .audio import get_melspectrogram_db, spec_to_image

from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm

class Audio(Dataset):
    def __init__(self, fnames, transform=None):
        self.data = []
        self.label = []
        
        for fname in tqdm(fnames):
            s = get_melspectrogram_db(fname)
            r = spec_to_image(s)
            r = np.expand_dims(r, axis=2).astype(np.float32)
            
            if transform:
                r = transform(r)
            self.data.append(r)
            
            label = int(fname.split("/")[1].split('.')[0].split('-')[1])
            self.label.append(label)


    def __getitem__(self, i):
        return self.data[i], self.label[i]

    
    def __len__(self):
        return len(self.data)