import os
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
import random

operating_system = 'linux'
split = '/'
def transfer_onehot_single(data):#变成独热向量
    frame, codeword = data.shape
    temp = np.zeros((frame, 3536))
    for f in range(frame):
        for c in range(codeword):
            index = int(data[f, c])
            temp = insert(temp, index, f, c)
    return temp


def get_codeword(idx, c):
    if c == 0:
        return idx
    elif c == 1:
        return idx + 32
    elif c == 2:
        return idx + 96
    elif c == 3:
        return idx + 128
    elif c == 4:
        return idx + 192
    elif c == 5:
        return idx + 448
    elif c == 6:
        return idx + 704
    elif c == 7:
        return idx + 960
    elif c == 8:
        return idx + 1216
    elif c == 9:
        return idx + 1472
    elif c == 10:
        return idx + 1474
    elif c == 11:
        return idx + 1476
    elif c == 12:
        return idx + 1478
    elif c == 13:
        return idx + 1480
    elif c == 14:
        return idx + 1736
    elif c == 15:
        return idx + 1992
    elif c == 16:
        return idx + 2248
    elif c == 17:
        return idx + 2504
    elif c == 18:
        return idx + 2508
    elif c == 19:
        return idx + 2636
    elif c == 20:
        return idx + 2640
    elif c == 21:
        return idx + 2768
    elif c == 22:
        return idx + 3024
    elif c == 23:
        return idx + 3280


def insert(temp, idx, frame, c):
    idx = get_codeword(idx, c)
    temp[frame, idx] = 1
    return temp



class ReadData(Dataset):
    def __init__(self, cover_dir, stego_dir, list):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.list = list
        self.cover_labels = np.array([0], dtype='int32')
        self.stego_labels = np.array([1], dtype='int32')
        assert len(self.list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        idx = int(idx)
        start, end, (file, label) = self.list[idx]
        if label == 0:
            path = os.path.join(self.cover_dir, file)
            label_array = self.cover_labels
        else:
            path = os.path.join(self.stego_dir, file)
            label_array = self.stego_labels

        cover = np.loadtxt(path)
        images = cover
        # images = cover[start:end]#若要测试时长，用这行代码改变时长
        images = transfer_onehot_single(images)

        samples = {'images': images, 'labels': label_array}

        return samples


class BalancedRandomData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * 2

    def __getitem__(self, index):
        reference_index = index // 2
        reference_reminder = index % 2
        current_data = self.dataset[reference_index]['images'][reference_reminder]
        current_label = np.array([self.dataset[reference_index]['labels'][reference_reminder]], dtype='int32')

        samples = {'images': current_data, 'labels': current_label}
        return samples


def read_data(cover_dir, stego_dir, shuffle, batch_size, num_workers, pin_memory):

    list_cover = [x.split(split)[-1] for x in glob(cover_dir + split + '*')]
    list_stego = [x.split(split)[-1] for x in glob(stego_dir + split + '*')]
    list_cover_labeled = [(file, 0) for file in list_cover]
    list_stego_labeled = [(file, 1) for file in list_stego]

    target_sec = 10 #修改音频时长

    list_cover_labeled = slice(target_sec, list_cover_labeled)
    list_stego_labeled = slice(target_sec, list_stego_labeled)
    list_cover_labeled.extend(list_stego_labeled)
    random.shuffle(list_cover_labeled)

    list_labeled = list_cover_labeled
    full_dataset = ReadData(cover_dir, stego_dir, list_labeled)

    sampler = RandomSampler(full_dataset)

    loader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    return loader



def slice(target_sec, list):
    file_length = (333//10)*target_sec
    file_num = int(333//file_length)
    start = 0
    end = int(start + file_length)
    temp = []
    for i in range(file_num):
        sliced = [(start, end, path_label) for path_label in list]
        temp.extend(sliced)
        start += int(file_length)
        end += int(file_length)
    return temp