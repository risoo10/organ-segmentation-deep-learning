import os
import numpy as np
import pydicom
import SimpleITK as sitk
from registration import *
from tqdm import tqdm


class DataEntry:
    def __init__(self, images, labels, z_positions):
        self.labels = None
        self.images = None
        self.z_positions = None


class PancreasDataset():
    def __init__(self):
        self.PANCREAS_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CT-PANCREAS\Pancreas-CT'
        self.PANCREAS_LABELS_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CT-PANCREAS\PANCREAS-LABELS'
        self.width = 512
        self.height = 512
        self.dataEntries = None

    def preprocess(self, ds):
        # Load slice index from Dicom
        slice_index = ds.data_element("InstanceNumber").value - 1
        img = normalize_CT(ds)
        return slice_index, img

    def load_by_id(self, PATIENT_ID):
        print(f'Loading Pancreas dataset for patient ID: {PATIENT_ID} ....')
        PATIENT_DIR = os.path.join(self.PANCREAS_DIR, f'PANCREAS_{PATIENT_ID}')
        dir1 = os.listdir(PATIENT_DIR)[0]
        dir2 = os.listdir(os.path.join(PATIENT_DIR, dir1))[0]
        final_dir = os.path.join(PATIENT_DIR, dir1, dir2)
        slices = os.listdir(final_dir)

        # Load and preprocess images
        images = np.zeros((len(slices) + 100, self.width, self.height))
        min_slice = 1000
        max_slice = 0
        for slice_number, slice_file in enumerate(slices):
            ds = pydicom.dcmread(os.path.join(final_dir, slice_file))
            slice_index, images[slice_index] = self.preprocess(ds)
            min_slice = slice_index if slice_index < min_slice else min_slice
            max_slice = slice_index if slice_index > max_slice else max_slice

        images = images[min_slice: max_slice + 1]

        # Load truth labels
        label_file = f'label{PATIENT_ID}.nii.gz'
        path = os.path.join(self.PANCREAS_LABELS_DIR, label_file)
        volume = sitk.ReadImage(path)
        labels = sitk.GetArrayFromImage(volume).astype(np.float64)

        print(f'Finished: images {images.shape}, labels {labels.shape}')

        return images, labels


class LiverDataset():
    def __init__(self):
        self.LIVER_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CHAOS_Train_Sets\Train_Sets\CT'
        self.IMAGES_DIR = 'DICOM_anon'
        self.LABELS_DIR = 'Ground'
        self.width = 512
        self.height = 512
        self.images = None
        self.labels = None
        self.z_positions = None
        self.registration = RigidRegistration(source=None, default_pixel_value=0)
        self.registration_source = None

    def set_source(self, source):
        self.registration_source = source
        self.registration.set_source(source)

    def preprocess(self, ds):
        # Load slice index from Dicom
        slice_orientation = ds.data_element("ImageOrientationPatient").value
        slice_location = ds.data_element("ImagePositionPatient").value
        # print(f'Slice position: {slice_location}, orientation: {slice_orientation}, device ser.n.: {ds.data_element("PatientPosition").value}')
        img = normalize_CT(ds)
        return img, slice_location

    def load_by_id(self, PATIENT_ID):
        print(f'Loading Liver dataset for patient: {PATIENT_ID} ....')

        image_path = os.path.join(self.LIVER_DIR, PATIENT_ID, self.IMAGES_DIR)
        image_files = os.listdir(image_path)

        label_path = os.path.join(self.LIVER_DIR, PATIENT_ID, self.LABELS_DIR)
        label_files = os.listdir(label_path)

        # Load and preprocess images
        assert len(label_files) == len(image_files)
        labels = np.zeros((len(label_files), self.width, self.height))
        images = np.zeros((len(image_files), self.width, self.height))
        z_positions = np.zeros((len(image_files)))
        for index, (image_file, label_file) in enumerate(zip(image_files, label_files)):

            # Load and preprocess images
            ds = pydicom.dcmread(os.path.join(image_path, image_file))
            image, slice_location = self.preprocess(ds)

            # Load and preprocess labels
            path = os.path.join(label_path, label_file)
            volume = sitk.ReadImage(path)
            label = sitk.GetArrayFromImage(volume).astype(np.float64)
            label = label / np.max(label)

            images[index] = image
            labels[index] = label
            # z_positions[index] = slice_location

        if self.registration_source is not None:
            # Apply Rigid tranfromation
            print('Applying rigid registration ...')
            self.registration.fit_transform(images[int(images.shape[0] / 2)])
            images = np.array([self.registration.transform_single(img) for img in tqdm(images)])
            labels = np.array([self.registration.transform_single(lbl) for lbl in tqdm(labels)])

        print(f'Finished: images {images.shape}, labels {labels.shape}, slice_coordinates min:{np.min(z_positions)}, max: {np.max(z_positions)}')
        return images, labels, z_positions

    def load_train(self):
        print(f'Loading Pancreas dataset (ALL) ....')

        # Load and set registration SOURCE
        images, labels, slice_coordinates = self.load_by_id('16')
        source = images[int(images.shape[0] / 2)]
        self.set_source(source)

        patient_ids = os.listdir(self.LIVER_DIR)
        print(patient_ids)
        # patient_ids = os.listdir(self.LIVER_DIR)[:2]  # EXAMPLE SIZE
        x_train = []
        y_train = []

        sum = 0
        for index, patient_id in enumerate(patient_ids):
            print(index, 'of', len(patient_ids))
            images, labels, _ = self.load_by_id(patient_id)
            x_train.append(images)
            y_train.append(labels)
            sum += images.shape[0]
            print(patient_id, ' - SUM:', sum)

            # patient_data[patient_id]['images'] = images
            # patient_data[patient_id]['labels'] = labels

        return np.concatenate(x_train), np.concatenate(y_train)


def normalize_CT(dicom_data):
    img = dicom_data.pixel_array.astype(np.int16)
    # Convert to Hounsfield units (HU)
    intercept = dicom_data.RescaleIntercept
    slope = dicom_data.RescaleSlope
    if slope != 1:
        img = slope * img.astype(np.float64)
        img = img.astype(np.int16)

    img += np.int16(intercept)
    img = np.array(img, dtype=np.int16)

    # Set outside-of-scan pixels to 0
    img[img < -2000] = intercept

    # Clip only HU of liver and tissues
    img = np.clip(img, -100, 300)

    # Normalize input
    copy = img
    min_, max_ = float(np.min(copy)), float(np.max(copy))
    img = (copy - min_) / (max_ - min_)

    return img
