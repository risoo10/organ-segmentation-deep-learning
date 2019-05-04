from data_loader import *
import tables

# Pancreas
# dataset = PancreasDataset()
# images, labels = dataset.load_by_id('0002')

train_sets = {}

# Liver
dataset = LiverDataset()
x_train, y_train = dataset.load_train()
print('Train:', 'x:', x_train.shape, 'y:', y_train.shape)

# Save to pytable
train_file = tables.open_file("CT-train.h5", mode="w")
x_atom = tables.Atom.from_dtype(x_train.dtype)
filters = tables.Filters(complib='blosc', complevel=5)
x_array = train_file.create_carray("/", "x", x_atom, x_train.shape, filters=filters)
y_atom = tables.Atom.from_dtype(y_train.dtype)
y_array = train_file.create_carray("/", "y", y_atom, y_train.shape, filters=filters)

# Write numpy array to files
x_array[:] = x_train
y_array[:] = y_train

train_file.close()




# for img, label, slice_coord in zip(images, labels, slice_coordinates):
#     print(f'Slice position: {slice_coord}')
#     transform = LocalBinaryPatterns()
#     # transform = GraphCut(img, label, dataset.width, dataset.height, showMask=True)
#     # transform = Thresholding()
#     transform.fit(img, label)


