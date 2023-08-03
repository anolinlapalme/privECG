from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import numpy as np
from ecgdetectors import Detectors
detectors = Detectors(50)

class DatasetTrain(Dataset):
    def __init__(self, X_data_train, y_data_train,r_data,non_r_data):
        self.x_data = X_data_train
        self.y_data =  y_data_train
        self.r_data = r_data
        self.non_r_data = non_r_data


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        return self.x_data[idx], self.y_data[idx], self.r_data[idx], self.non_r_data[idx]


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_all(data):
    for row in tqdm(range(data.shape[0])):
        for col in range(data.shape[1]):
            data[row][col] = np.squeeze(NormalizeData(data[row][col])).astype('float32')
    return data


def make_masks(dataset):

    list_data_R_wave = list()
    list_data_non_R_wave = list()
    for ECG in tqdm(dataset):
        r_peaks = detectors.christov_detector(ECG[1]) #take lead 1

        l_empty = list()
        for i in r_peaks:
            l_empty += list(range(i-5,i+6))

        list_dummy = np.asarray([0] * 500)

        list_dummy[l_empty] = [1] * len(l_empty)

        list_data_R_wave.append(list_dummy)
        list_data_non_R_wave.append(np.invert(list_dummy))

    return np.array(list_data_R_wave), np.array(list_data_non_R_wave)


