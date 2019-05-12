from data_loader import *
import tables
from utils import *
from thresholding import *
from graph_cut import *


train_sets = {}

#######################
# Export to Pytables
#######################
dataset = LiverDataset()
x, y = dataset.load_set("train")
print('Test:', 'x:', x.shape, 'y:', y.shape)

# Save to pytable
test_file = tables.open_file("CT-test.h5", mode="w")
x_atom = tables.Atom.from_dtype(x.dtype)
filters = tables.Filters(complib='blosc', complevel=5)
x_array = test_file.create_carray("/", "x", x_atom, x.shape, filters=filters)
y_atom = tables.Atom.from_dtype(y.dtype)
y_array = test_file.create_carray("/", "y", y_atom, y.shape, filters=filters)

# Write numpy array to files
x_array[:] = x
y_array[:] = y
test_file.close()

#######################
# Export to video
#######################
# dataset = LiverDataset()
# # Load and set registration SOURCE
# images, labels, slice_coordinates = dataset.load_by_id('16')
# var_images, var_labels, var_slice_coordinates = dataset.load_by_id('23')
# export_video(images, labels, 512, 512, 'plots/liver-slice')


#######################
# Thresholding
#######################
# thresholding = Thresholding()
# thresholding.fit(images[50], labels[50])
# img = images[50]
# label = labels[50]


# graph_cut = GraphCut(img, label, 512, 512)
# graph_cut.fit(img, label)

#######################
# REGISTRATION SHOWCASE
#######################
# img = images[images.shape[0] // 2]
# var_img = var_images[var_images.shape[0] // 2]
# registration = RigidRegistration(img, 1.0)
# registration.set_source(img)
# registration.fit_transform(var_img)
# target = registration.transform_single(var_img)

# figure = plt.figure(figsize=(10, 15))
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap="bone")
# plt.subplot(1, 3, 2)
# plt.imshow(var_img, cmap="bone")
# plt.subplot(1, 3, 3)
# plt.imshow(target, cmap="bone")
# plt.show()


#######################
# Local binary patterns
#######################
# for img, label, slice_coord in zip(images, labels, slice_coordinates):
#     print(f'Slice position: {slice_coord}')
#     transform = LocalBinaryPatterns()
#     # transform = GraphCut(img, label, dataset.width, dataset.height, showMask=True)
#     # transform = Thresholding()
#     transform.fit(img, label)


