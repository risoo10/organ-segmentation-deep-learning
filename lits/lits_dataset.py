import tables
import numpy as np
from utils.constants import *


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

    def load_write(self):
        self.file = tables.open_file(f'{drive_dir}/{self.filename}', mode="w")
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
        train, test, val = self.split_test_val_train(PATIENTS)
        self.file.create_array('/', 'train', train)
        self.file.create_array('/', 'test', test)
        self.file.create_array('/', 'val', val)

    def write_cropped_slices(self, slcs):
        self.file.create_array('/', 'cropped_slices', slcs)
        self.cropped_slices = slcs

    def load(self, mode):
        self.file = tables.open_file(f'{drive_dir}/{self.filename}', mode=mode)
        self.opened = True
        self.x = self.file.get_node('/x')
        self.y = self.file.get_node('/y')
        self.slices = self.file.get_node('/slices')

        try:
            self.train = self.file.get_node('/train')
            self.test = self.file.get_node('/test')
            self.val = self.file.get_node('/val')
            self.cropped_slices = self.file.get_node('/cropped_slices')
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

    def close(self):
        self.opened = False
        self.file.close()
        del self.x
        del self.y
        del self.slices
