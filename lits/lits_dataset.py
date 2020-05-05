import tables
import numpy as np
from utils.constants import *
import pickle


class LitsDataSet():
    def __init__(self, filename):
        self.opened = False
        self.filename = filename
        self.file = None
        self.x = None
        self.y = None
        self.slices = None
        self.cropped_slices = None
        self.train = None
        self.val = None
        self.test = None
        self.liver_detected = None

    def load_write(self):
        self.file = tables.open_file(f'{drive_dir}/{self.filename}.h5', mode="w")
        self.opened = True

        filters = tables.Filters(complib='blosc', complevel=5)
        x_atom = tables.Atom.from_dtype(np.dtype('float16'))
        y_atom = tables.Atom.from_dtype(np.dtype('int8'))
        slices_atom = tables.Atom.from_dtype(np.dtype('int32'))

        self.x = self.file.create_carray(
            "/", 'x', x_atom, (SAMPLES, HEIGHT, WIDTH), filters=filters)
        self.y = self.file.create_carray(
            "/", 'y', y_atom, (SAMPLES, HEIGHT, WIDTH), filters=filters)
        self.slices = self.file.create_carray(
            '/', 'slices', slices_atom, (PATIENTS, 2), filters=filters)

    def write_splits(self):
        self.train, self.test, self.val = self.split_test_val_train(PATIENTS)
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
        data["liver_detected"] = self.liver_detected        
        pickle.dump(data, open(f'{drive_dir}/{self.filename}.p', "wb"))

    def load(self, mode):
        self.file = tables.open_file(f'{drive_dir}/{self.filename}.h5', mode=mode)
        self.opened = True
        self.x = self.file.get_node('/x')
        self.y = self.file.get_node('/y')
        self.slices = self.file.get_node('/slices')

        try:
            data = pickle.load(open(f'{drive_dir}/{self.filename}.p', "rb"))
            self.train = data['train']
            self.test = data['test']
            self.val = data['val']
            self.cropped_slices = data['cropped_slices']
            self.liver_detected = data['liver_detected']
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

    def get_weights(self):
        total = len(self.x)
        liver_detected = self.liver_detected.astype(np.bool)
        positive = liver_detected.sum()
        negative = total - positive
        pos_w = 1 / positive
        neg_w = 1 / negative
        weights = np.zeros(liver_detected.shape)
        weights[liver_detected] = pos_w
        weights[np.invert(liver_detected)] = neg_w 
        return weights

    def close(self):
        self.opened = False
        self.file.close()
        del self.x
        del self.y
        del self.slices
