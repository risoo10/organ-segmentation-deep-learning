# Liver Segmentation from Radiological Imaging (CT)

## Dataset
DATASET 
For this segmentation task we used radiological imaging dataset published in **CHAOS Open Grand Challenge** in 2018. Data were obtained from 40 healthy patients using Computed Tomography and liver was manually annotated by expert radiologist for each slice. Image sets were not registered.

## Evaluation
We used **Dice-Sorensen Coefficient (DSC)** for evaluation of each method.  Coefficient is defined as intersection over union.   

## Segmentation Methods:

### 1. Thresholding
We applied thresholding to obtain segmentation mask using Binary Thresholding and OTSU Adaptive Thresholding. We used interactive selection of the threshold value for the Binary Thresholding method. DSC evaluation of both methods was respectively DSC(*binary*) ≈ 0,4012 , DSC(*otsu*) ≈ 0,3974.

### 2. Graph cut
We used interactive method of selecting liver - foreground and background parts of CT image and then we applied *Graph Cut* energy based algorithm to obtain segmentation mask. Foreground and background parts could be interatively updated to recalculate mask to obtain more precise segmentation. Results can be seen on Figure 1. 


![Figure 1.: Graph Cut - interactive segmentation method](https://github.com/vgg-fiit/pv-semestralny-projekt-organ-segmentation-mocak/blob/master/plots/graphcut.gif?raw=true)


### 3. Deep Convolutional Neural Network

#### Rigid Registration
We applied Rigid Regsitration technique to eliminate different between two patient's angles and  scaling and centering differences. Registration used Stochastic Gradient Descent optimization of the transformation using operations such as rotation, semantic scaling and shifting. We provided method with reference image and target image to apply registration. Example can be seen on Figure 2.

![Figure 2.: Rigid Registration](https://github.com/vgg-fiit/pv-semestralny-projekt-organ-segmentation-mocak/blob/master/plots/Rigid%20Registration.png?raw=true)

#### Data Augmentation
We applied two 

#### U - Net 
We applied two 
