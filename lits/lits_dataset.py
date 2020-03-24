import tables
import numpy as np
from utils.constants import *

class LitsDataSet:
  def __init__(self, filename):
    self.opened = False
    self.filename = filename
    self.file = None
    self.x = None
    self.y = None
    self.slices = None

  def load_write(self):
    self.file = tables.open_file(f'{drive_dir}/{self.filename}', mode="w")
    self.opened = True

    filters = tables.Filters(complib='blosc', complevel=5)
    x_atom = tables.Atom.from_dtype(np.dtype('float16'))
    y_atom = tables.Atom.from_dtype(np.dtype('int8'))
    slices_atom = tables.Atom.from_dtype(np.dtype('int16'))

    self.x = self.file.create_carray("/", 'x', x_atom, (SAMPLES, HEIGHT, WIDTH), filters=filters)
    self.y = self.file.create_carray("/", 'y', y_atom, (SAMPLES, HEIGHT, WIDTH), filters=filters)
    self.slices = self.file.create_carray('/', 'slices', slices_atom, (PATIENTS, 2), filters=filters)

  def load_read(self):
    self.file = tables.open_file(f'{drive_dir}/{self.filename}', mode="r")
    self.opened = True
    self.x = self.file.get_node('/x')
    self.y = self.file.get_node('/y')
    self.slices = self.file.get_node('/slices')

  def save(self, x, y, start, end, patient_ind):
    self.x[start:end] = x
    self.y[start:end] = y
    self.slices[patient_ind] = np.array([start, end]).astype(np.int16)

  def close(self):
    self.opened = False
    self.file.close()
    del self.x
    del self.y
    del self.slices