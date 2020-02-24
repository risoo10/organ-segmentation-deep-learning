import SimpleITK as sitk
from computer_vision.data_loader import *
import matplotlib.pyplot as plt


class RigidRegistration():
    def __init__(self, source, default_pixel_value):
        self.source = source
        self.source_img = None
        self.resampler = None
        self.iterations = 100
        self.default_value = default_pixel_value

    def set_source(self, source):
        self.source = source
        self.source_img = sitk.GetImageFromArray(source)

    # Code was modified from external source at official page of SimpleITK library
    # Code was introduces as sample for Rigid Transformation in jupyter notebooks at
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/
    def fit_transform(self, target):

        target_img = sitk.GetImageFromArray(target)

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=self.iterations,
                                                   gradientMagnitudeTolerance=1e-8)
        R.SetOptimizerScalesFromIndexShift()

        tx = sitk.CenteredTransformInitializer(self.source_img, target_img, sitk.Similarity2DTransform())
        R.SetInitialTransform(tx)

        out_tx = R.Execute(self.source_img, target_img)

        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetReferenceImage(self.source_img)
        self.resampler.SetInterpolator(sitk.sitkLinear)
        self.resampler.SetDefaultPixelValue(self.default_value)
        self.resampler.SetTransform(out_tx)
        out = self.resampler.Execute(target_img)

        return sitk.GetArrayFromImage(out)

    def transform_single(self, target):
        assert self.resampler is not None
        assert self.source_img is not None

        target_img = sitk.GetImageFromArray(target)
        out = self.resampler.Execute(target_img)
        return sitk.GetArrayFromImage(out)

    def plot_example(self, target, transformed):
        fig = plt.figure(figsize=(13, 8))
        fig.add_subplot(1, 3, 1)
        plt.imshow(self.source, cmap="bone")
        plt.title("Source")
        fig.add_subplot(1, 3, 2)
        plt.imshow(target, cmap="bone")
        plt.title("Target")
        fig.add_subplot(1, 3, 3)
        plt.imshow(transformed, cmap="bone")
        plt.title("Transformed")

        plt.show()
