import h5py
from utils.constants import *
import numpy as np
import pickle


class DynamicDataset():
    def __init__(self, filepath, shape, sequences):
        self.opened = False
        self.filename = filepath
        self.file = None
        self.x = None
        self.y = None
        self.slices = None
        self.cropped_slices = None
        self.train = None
        self.val = None
        self.test = None
        self.shape = shape
        self.sequences = sequences

    def load_write(self):
        self.file = h5py.File(f'{drive_dir}/{self.filename}.h5', mode="w")
        self.opened = True

        self.x = self.file.create_dataset(
            'x', shape=self.shape, dtype=np.float16, compression='gzip')
        self.y = self.file.create_dataset(
            'y', shape=self.shape, dtype=np.int8, compression='gzip')
        self.slices = self.file.create_dataset(
            'slices', shape=(self.sequences, 2), dtype=np.int8, compression='gzip')

    def write_splits(self):
        self.train, self.test, self.val = self.split_test_val_train(
            self.sequences)
        self.write_pickle_data()

    def write_cropped_slices(self, slcs):
        self.cropped_slices = slcs
        self.write_pickle_data()

    def write_pickle_data(self):
        data = {}
        data["train"] = self.train
        data["test"] = self.test
        data["val"] = self.val
        data["cropped_slices"] = self.cropped_slices
        pickle.dump(data, open(f'{drive_dir}/{self.filename}.p', "wb"))

    def load(self, mode):
        self.file = h5py.File(
            f'{drive_dir}/{self.filename}.h5', mode=mode)
        self.opened = True
        self.x = self.file['x']
        self.y = self.file['y']
        self.slices = self.file['slices']

        try:
            data = pickle.load(open(f'{drive_dir}/{self.filename}.p', "rb"))
            self.train = data['train']
            self.test = data['test']
            self.val = data['val']
            self.cropped_slices = data['cropped_slices']
        except:
            print('Tran or Test or Val or Cropped slices not set')

    def save(self, x, y, start, end, patient_ind):
        self.x[start:end] = x
        self.y[start:end] = y
        self.slices[patient_ind] = np.array([start, end]).astype(np.int32)

    def split_test_val_train(self, length):
        indices = np.arange(0, length)
        np.random.seed(42)
        np.random.shuffle(indices)
        splits = [int(length * 0.7), int(length * 0.9)]
        return np.split(indices, splits)
