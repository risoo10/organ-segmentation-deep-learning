# Liver Segmentation from Radiological Imaging (CT)

## Dataset
DATASET 
For this segmentation task we used radiological imaging dataset published in **CHAOS Open Grand Challenge** in 2018. Data were obtained from 40 healthy patients using Computed Tomography and liver was manually annotated by expert radiologist for each slice. Image sets were not registered.

## Evaluation
We used **Dice-Sorensen Coefficient (DSC)** for evaluation of each method.  Coefficient is defined as intersection over union. For evaluation of the neural network method we used also standard classification metrics Precision and Recall. 

## Segmentation Methods:

### 1. Thresholding
We applied thresholding to obtain segmentation mask using Binary Thresholding and OTSU Adaptive Thresholding. We used interactive selection of the threshold value for the Binary Thresholding method. DSC evaluation of both methods was respectively DSC(*binary*) ≈ 0,4012 , DSC(*otsu*) ≈ 0,3974.

### 2. Graph cut
We used interactive method of selecting liver - foreground and background parts of CT image and then we applied *Graph Cut* energy based algorithm to obtain segmentation mask. Foreground and background parts could be interatively updated to recalculate mask to obtain more precise segmentation with DSC(*graphcut*) ≈ 0,8132). Results can be seen on Figure 1. 


![Figure 1.: Graph Cut - interactive segmentation method](https://github.com/vgg-fiit/pv-semestralny-projekt-organ-segmentation-mocak/blob/master/plots/graphcut.gif?raw=true)


### 3. Deep Convolutional Neural Network

#### Rigid Registration
We applied Rigid Regsitration technique to eliminate differences between two patient's angles and  scaling and centering differences. Registration used Stochastic Gradient Descent optimization of the applied transformation using operations such as rotation, semantic scaling and shifting. We provided method with reference image and target image to apply registration. Example can be seen on Figure 2.

![Figure 2.: Rigid Registration](https://github.com/vgg-fiit/pv-semestralny-projekt-organ-segmentation-mocak/blob/master/plots/Patient%20Variance%20Registration%20(RIGID).png?raw=true)

#### Data Augmentation
We used data augmentation approaches to add variance to training dataset and to prevent overfitting of the network and lead to better generalization of the algorithm. We only used subtle augmentations using random rotation, semantic scaling and shifting operation with probability of 0.8. 

#### U - Net 
We trained Deep Convolutional Neural Network model called U-net. U-net architecture consists of Convolutional layers with RELU activation function and Max Pooling layers. Architecture of the network brings into attention mechanism for concatenation of so called Skipped connections. 

We trained this model for 50 epochs using mini batch aproach with batch size of 6. DSC was used as loss function and Adam optimizer was used to update weights in network. 

We evaluated model on the test set using mean and median  of the DSC, Precision and Recall metrics. Results can be seen in Table 1. Example of segementation using U-net neural network model can be seen in Figure 3. 

|          |      DSC      |  Precision    |   Recall      |
|:---------|:-------------:|:-------------:|:-------------:|
| mean     |    0.8167     |    0.7632     |    0.7566     |
| median   |    0.9659     |    0.9476     |    0.9843     |
Table 1.: Evaluation of the U-net neural network model


![Figure 3.: U-net segementaion example (DSC is 0,9815)](https://github.com/vgg-fiit/pv-semestralny-projekt-organ-segmentation-mocak/blob/master/plots/Patient%20Variance%20Registration%20(RIGID).png?raw=true)



