import os
import random
# import pydicom
from model import *
import numpy as np
import cv2
# from tqdm import tqdm
# import SimpleITK as sitk
import matplotlib.pyplot as plt
import tables
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate
)
from keras.callbacks import TensorBoard

COLUMNS = 512
ROWS = 512
MODEL = 'unet-REG-AUG'
main_dir = ".."

## Open HDF FILE in READ mode
train_file = tables.open_file(f'../CT-train.h5', mode="r")
print(train_file)

x_train = train_file.get_node('/x')
y_train = train_file.get_node('/y')

## AUGMENTATION pipeline

TRAIN_AUGMENT = Compose([
    # VerticalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.8),
])


def generator(X_data, y_data, batch_size, indices, augmentation=None):
    indx = indices
    samples_per_epoch = len(indx)
    number_of_batches = samples_per_epoch / batch_size
    counter = 0

    while 1:
        start = batch_size * counter
        stop = batch_size * (counter + 1)
        indxs = indx[start:stop]
        #     print(start, stop, indxs)

        X_batch = np.zeros((batch_size, X_data.shape[1], X_data.shape[2]))
        y_batch = np.zeros((batch_size, y_data.shape[1], y_data.shape[2]))

        for i, idx in enumerate(indxs):
            if augmentation is not None:
                aug = augmentation(image=X_data[idx], mask=y_data[idx])
                X_batch[i] = aug["image"]
                y_batch[i] = aug["mask"]
            else:
                X_batch[i] = X_data[idx]
                y_batch[i] = y_data[idx]

        counter += 1
        shape = X_batch.shape
        #     print('Batch X, y:', X_batch.shape, y_batch.shape)
        yield X_batch.reshape((-1, shape[1], shape[2], 1)), y_batch.reshape((-1, shape[1], shape[2], 1))

        # restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            np.random.shuffle(indx)
            counter = 0


## SPLIT TRAIN AND VALIDATION SETS

TRAIN_SIZE = x_train.shape[0]
indices = np.arange(TRAIN_SIZE)
# np.random.shuffle(indices)
split = 2302
train_ind = indices[:split]
eval_ind = indices[split:]

print("Train IND (example):", train_ind[:10], ', shape:', train_ind.shape)
print("Validation IND (example):", eval_ind[:10], ', shape:', eval_ind.shape)


## Train MODEL
model = u_net((COLUMNS, ROWS, 1))
model.summary()

model_checkpoint = ModelCheckpoint(f'{main_dir}/models/checkpoint/{MODEL}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
tbCallBack = TensorBoard(log_dir=f'{main_dir}/logs', histogram_freq=0, write_graph=True, write_images=True)

epochs = 50
batch_size = 12

# No augmentation
train_gen = generator(x_train, y_train, batch_size, train_ind)
eval_gen = generator(x_train, y_train, batch_size, eval_ind)

# WITH AUGMENTATION
# train_gen = generator(x_train, y_train, batch_size, train_ind, augmentation=TRAIN_AUGMENT)
# eval_gen = generator(x_train, y_train, batch_size, eval_ind, augmentation=TRAIN_AUGMENT)


history = model.fit_generator(
    generator=train_gen,
    epochs=epochs,
    verbose=1,
    steps_per_epoch=int(train_ind.shape[0] / batch_size),
    validation_data=eval_gen,
    validation_steps=int(eval_ind.shape[0] / batch_size),
    callbacks=[model_checkpoint, tbCallBack])

# Save model and tensor logs
model.save(f'{main_dir}/models/{MODEL}.hdf5')







