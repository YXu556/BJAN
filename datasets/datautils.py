"""
This script is for data augmentation
"""
import random
import numpy as np

import torch
from torch.utils.data import ConcatDataset



# MEAN = {
#     '2019': np.array([0.15, 0.173, 0.19, 0.227, 0.286, 0.31, 0.322, 0.33, 0.266, 0.194]),
#     '2020': np.array([0.095, 0.121, 0.143, 0.179, 0.244, 0.274, 0.288, 0.3, 0.29, 0.206]),
#     '2021': np.array([0.102, 0.127, 0.147, 0.183, 0.245, 0.275, 0.289, 0.3, 0.291, 0.209]),
#     '2022': np.array([0.177, 0.202, 0.227, 0.26, 0.316, 0.345, 0.359, 0.371, 0.4, 0.323])
# }
# STD = {
#     '2019': np.array([0.232, 0.224, 0.226, 0.223, 0.201, 0.193, 0.192, 0.18, 0.119, 0.103]),
#     '2020': np.array([0.12, 0.116, 0.126, 0.125, 0.124, 0.133, 0.133, 0.132, 0.109, 0.097]),
#     '2021': np.array([0.135, 0.13, 0.137, 0.135, 0.129, 0.134, 0.135, 0.132, 0.107, 0.098]),
#     '2022': np.array([0.086, 0.085, 0.097, 0.094, 0.097, 0.11, 0.108, 0.109, 0.114, 0.11])
# }
# ---------------------------- Spectral augmentation ----------------------------
class RandomAddNoise:
    def __call__(self, sample):
        x = sample['x']
        t, c = x.shape
        for i in range(t):
            prob = np.random.rand()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.5:
                    x[i, :] += -np.abs(np.random.randn(c)*0.5)  # np.random.uniform(low=-0.5, high=0, size=(c,))
                else:
                    x[i, :] += np.abs(np.random.randn(c)*0.5)  # np.random.uniform(low=0, high=0.5, size=(c,))
        sample['x'] = x
        return sample


# ---------------------------- Temporal augmentation ----------------------------
class RandomTempShift:
    def __init__(self, max_shift=30, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, sample):
        p = np.random.rand()
        doy = sample['doy']
        if p < self.p:
            # t_shifts = random.randint(-self.max_shift, self.max_shift)
            # sample['x'] = np.roll(sample['x'], t_shifts, axis=0)
            shift = np.clip(np.random.randn()*0.3, -1, 1) * self.max_shift  # random.randint(-self.max_shift, self.max_shift)
            doy = doy + shift
        sample['doy'] = doy
        return sample


class RandomTempRemoval:
    def __call__(self, sample):
        x = sample['x']
        doy = sample['doy']
        mask = [1 if random.random() < 0.15 else 0 for _ in range(x.shape[0])]
        mask = np.array(mask) == 0
        sample['x'] = x[mask]
        sample['doy'] = doy[mask]

        return sample


# -------------------------- Data process ---------------------- #
class RemoveNoise:
    def __init__(self, year):
        self.year = year

    def isvalid(self, x):
        score = np.ones(x.shape[0])
        score = np.minimum(score, (x[:, 0] - 0.1) / 0.4)  # blue
        score = np.minimum(score, (x[:, [0, 1, 2]].sum(1) - 0.2) / 0.6)  # rgb
        ndmi = (x[:, 6] - x[:, 8]) / (x[:, 6] + x[:, 8] + 1e-8)
        score = np.minimum(score, (ndmi + 0.1) / 0.2)  # ndmi
        cloud = score * 100 > 10

        dark = x[:, [6, 8, 9]].sum(1) < 0.35

        invalid = np.any(x == 0, 1)

        return ~(dark | cloud | invalid)

    def __call__(self, sample):
        x = sample['x']
        doy = sample['doy']

        if self.year == '2022':
            norm_x = (x[:, :10] - 1000) / 10000
        else:
            norm_x = x[:, :10] / 10000

        valid = self.isvalid(norm_x)

        sample['x'] = x[valid]
        sample['doy'] = doy[valid]

        return sample


class RandomSampleTimeSteps:
    def __init__(self, sequencelength, inseason=0, rc=False, rf=False):
        self.sequencelength = sequencelength
        self.rc = rc
        self.rf = rf
        if inseason == 0:
            self.inseason = None
        else:
            self.inseason = inseason

    def __call__(self, sample):
        x = sample['x']
        doy = sample['doy']
        if self.inseason:
            valid = doy <= self.inseason
            x = x[valid]
            doy = doy[valid]

        if self.rf:  # if rf is True, composite
            doy_pad = np.linspace(0, 366, self.sequencelength).astype('int')
            x_pad = np.array([np.interp(doy_pad, doy, x[:, i]) for i in range(10)]).T
            mask = np.ones((self.sequencelength,), dtype=int)
        elif self.rc:
            # choose with replacement if sequencelength smaller als choose_t
            replace = False if x.shape[0] >= self.sequencelength else True
            idxs = np.random.choice(x.shape[0], self.sequencelength, replace=replace)
            idxs.sort()

            # must have x_pad, mask, and doy_pad
            x_pad = x[idxs]
            mask = np.ones((self.sequencelength,), dtype=int)
            doy_pad = doy[idxs]
        else:
            # padding
            x_length, c_length = x.shape

            if x_length <= self.sequencelength:
                mask = np.zeros((self.sequencelength,), dtype=int)
                mask[:x_length] = 1

                x_pad = np.zeros((self.sequencelength, c_length))
                x_pad[:x_length, :] = x[:x_length, :]

                doy_pad = np.zeros((self.sequencelength,), dtype=int)
                doy_pad[:x_length] = doy[:x_length]
            else:
                idxs = np.random.choice(x.shape[0], self.sequencelength, replace=False)
                idxs.sort()

                x_pad = x[idxs]
                mask = np.ones((self.sequencelength,), dtype=int)
                doy_pad = doy[idxs]
        sample['x'] = x_pad
        sample['doy'] = doy_pad
        sample['mask'] = mask == 0

        return sample


class Normalize:
    def __init__(self, year):
        self.year = year
        self.mean = np.array([[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]])
        self.std = np.array([0.227, 0.219, 0.222, 0.22 , 0.2  , 0.193, 0.192, 0.182, 0.123, 0.106])

    def __call__(self, sample):
        x = sample['x']
        x = x[:, :10] * 1e-4
        if self.year == '2022':
            x -= 0.1
        x = (x - self.mean) / self.std
        sample['x'] = x
        return sample


class ToTensor:
    def __call__(self, sample):
        sample['x'] = torch.from_numpy(sample['x']).type(torch.FloatTensor)
        sample['mask'] = torch.from_numpy(sample['mask'])
        sample['doy'] = torch.from_numpy(sample['doy']).type(torch.LongTensor)
        sample['y'] = torch.tensor(sample['y'], dtype=torch.long)

        return sample


# -------------------------- Dataset utils ------------------------ #
class CropConcatDataset(ConcatDataset):
    r"""
    Inherits from:
    ConcatDataset - Dataset as a concatenation of multiple datasets.

    Add attribute from `USCrops`.
    """
    def __init__(self, datasets):
        super(CropConcatDataset, self).__init__(datasets)

        self.X_list = []
        for d in self.datasets:
            self.X_list.extend(d.X_list)
        # self.X_list = np.concatenate(X_list)

        y_list = []
        for d in self.datasets:
            y_list.extend(d.y_list)
        self.y_list = np.array(y_list)

    def update_transform(self, transform):
        for d in self.datasets:
            d.transform = transform


class Subset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.y_list = dataset.y_list[indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def update_transform(self, transform):
        self.dataset.update_transform(transform)


# -------------------------- Dataloader utils ------------------------ #
class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, batch_size):
        classes = sorted(set(labels))
        print(classes)

        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in classes
        ]

        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]
