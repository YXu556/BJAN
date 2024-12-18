'''
Todo:
    *
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset


class USCrops(Dataset):
    def __init__(self,
                 name,
                 root,
                 year=None,
                 fold=None,
                 mapblock=False,
                 indices=None,
                 preload_ram=True,
                 transform=None
                 ):
        super(USCrops, self).__init__()

        self.root = root
        if 'All' in name or 'all' in name:
            if len(name.split('_')) == 2:
                site, year = name.split('_')
                self.cache = root / str(year) / 'npy' / f"{site}_F{fold}.npy"
                self.indexfile = root / str(year) / 'indices' / f'{site}.csv'
            else:
                site, per, year = name.split('_')
                self.cache = root / str(year) / 'npy' / f"{site}_{per}_F{fold}.npy"
                self.indexfile = root / str(year) / 'indices' / f'{site}_{per}.csv'
        elif name.isdigit():
            self.cache = root / str(year) / 'cropmap' / 'npy' / f"Block{name}.npy"
            self.indexfile = root / str(year) / 'cropmap' / f'block_index.csv'
            mapblock = True
        elif len(name.split('_')[-1]) != 4:
            site, _ = name.split('_')
            self.cache = root / str(year) / 'npy' / site / f"{name}.npy"
            self.indexfile = root / str(year) / 'cropmap' / f'{name}.csv'
        else:
            site, year = name.split('_')
            self.cache = root / str(year) / 'npy' / site / f"{site}_{year}_F{fold}.npy"
            self.indexfile = root / str(year) / 'indices' / f'{site}.csv'

        # index
        self.index = pd.read_csv(self.indexfile, index_col=None)
        if fold:
            self.index = self.index[self.index.fold == fold]
        if mapblock:
            self.index = self.index[self.index.blockid == int(name)]

        self.index = self.index.loc[self.index.sequencelength > 10]#.set_index("idx")

        # cache
        if preload_ram:
            # print('Load', name, 'dataset')
            if self.cache.exists():
                self.load_cached_dataset()
            else:
                self.cache_dataset()
        else:
            # print('Load', name, 'dataset while training')
            self.X_list = None

        if indices:
            self.index = self.index[indices]
            self.X_list = self.X_list[indices]
            self.y_list = self.y_list[indices]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def load_cached_dataset(self):
        print("precached dataset files found at " + str(self.cache))
        self.data = np.load(self.cache, allow_pickle=True)
        self.X_list = self.data[0].tolist()
        self.y_list = self.data[1].astype('int64')
        if len(self.X_list) != len(self.index):
            print("cached dataset do not match. iterating through csv files.")
            self.cache_dataset()

    def cache_dataset(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.X_list = list()
        for idx, row in tqdm(self.index.iterrows(), desc="loading data into RAM", total=len(self.index)):
            self.X_list.append(pd.read_csv(row.path).values)
        self.y_list = self.index.classid.values
        np.save(self.cache, np.vstack([np.array(self.X_list), self.y_list]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]
        y = self.y_list[index]

        if self.X_list is None:
            X = pd.read_csv(row.path).values
        else:
            X = self.X_list[index]

        sample = {
            'x': X[:, :10],
            'doy': X[:, -1],
            'y': y
        }

        if self.transform:
            sample = self.transform(sample)

        X = [sample['x'], sample['mask'], sample['doy']]
        y = sample['y']

        return X, y


class PartialUSCrops(USCrops):
    def __init__(self, partial_classes, **kwargs):
        super(PartialUSCrops, self).__init__(**kwargs)
        assert all([c in self.classes for c in partial_classes])
        samples = []
        for (path, label) in self.samples:
            class_name = self.classes[label]
            if class_name in partial_classes:
                samples.append((path, label))
        self.samples = samples
        self.partial_classes = partial_classes
        self.partial_classes_idx = [self.class_to_idx[c] for c in partial_classes]


if __name__ == '__main__':
    dataset = USCrops(mode='train',
                      root=Path(r'D:\DadaX\PhD\Research\2. PheCo\data\US'))
    print(dataset[0])
