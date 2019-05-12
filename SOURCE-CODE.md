# Repository architecture
Source code is split into separate files for better readability. 

- File `data_loader.py`: incl. DICOM images loading, normalization
- File `graph_cut.py` : incl. intercative graph cut method using OpenCV library
- File `local_binary_patterns.py` : incl. local binary patterns using scikit-image library
- File `metrics.py` : incl. Dice-sorensen coefficient metric
- File `registration.py` : incl. Rigid Registration using SimpleITK library
- File `thresholding.py` : incl. Binary and Adaptive thresholding using OpenCV library
- File `utils.py` : incl. Utility functions
- File `main.py` - entry point to run methods

Actual training of the U-net method was implemented using Google Colab and Jupyter notebooks.

- Notebook `notebooks/U-net-training.ipynb` - Includes U-net model, Augmentation configuration, Model training and Evaluation.
 
## Source code originality and references
These parts of source code were copied and modified from internet:

- `graph_cut.py` - Interactive method of Graph Cut was modified from original OpenCV repository of samples at [github.com](https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py).

- `notebooks/U-net-training.ipynb` U-net implementation in Keras was modified from this source at  [github.com](https://github.com/zhixuhao/unet/blob/master/model.py).

- `registration.py`: Code for Rigid Registration was modified from external source at official page of SimpleITK library. Code was introduces as sample for Rigid Transformation in jupyter notebooks at [github.io](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/). 

All other parts of source do not include any copied or modified code from other sources.

