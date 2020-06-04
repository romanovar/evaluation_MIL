# Reproducibility of 'Label stability'

This is a short introduction into the project structure and reproducibility.

<!-- **Contents**
* Getting started
* Project structure

Introduction structure -->
 To reproduce the results of the project 3 important aspects are important:
  * virtual environment
  * script parameters
  * project modules

We will look into each one of them in details.

## Getting started - Requirements
Requirements for this projects are:
- [x] Python version 3.6.x
- [x] Requirements.txt lists all the necessary libraries needed for the project and their versions. 
[More about virtual environments and how to install packages from requirements.txt](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
- [x] CUDA version 10.2.89
- [x] Ubuntu 18.04.4
- [x] Nvidia driver version 440.82
- [x] GPU GeForce GTX 1080 Ti
 
 ##Datasets
We use 3 data sets, two of which are medical, and one non-medical:
 * [chest X-ray dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) [1], for short referred as X-Ray in the parameters file.
 * [MURA dataset](https://arxiv.org/abs/1712.06957) [2], for short referred as MURA
 * [Pascal VOC 2005](http://host.robots.ox.ac.uk/pascal/VOC/voc2005/index.html), for short referred as Pascal
 
[1]: Wang, Xiaosong et al. “ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017).

[2]: Rajpurkar, Pranav et al. “MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs.” ArXiv abs/1712.06957 (2017).


## Modules
Project is divided into 2 modules - `cnn` and `stability`.

#### CNN
`cnn` modules contains all scripts, which are in some way related to the neural architecture - this includes training,
 evaluating performance of a model, preprocessing
 Scripts module contains the following scripts:
* `train_model.py` allows explorative training of models.

* `run_cross_validation.py` performs cross validation on a specific architecture. In xray, every split is yields different train/validation/testing set. In Mura, the testing set is fixed, it is equal to the validation set from the author split.

* `train_models_on_subset.py` trains models on training subset, which are later tested for stability. Initially, the script takes a specific cross validation split of training, validation and testing set, and then drops a portion of the samples from the training set.    

* `evaluate_performance.py` evaluates the performance of a trained model. It is always calculated AUC for the set. If segmentation labels are available - it is calculated the dice coefficient and accuracy from IOU (with threshold of 0.1).
*
`train_mode` parameter in the settings is applied in these 3 script, the user specify if training should occur or prediction on already trained model. The model name should be specified in the script. 

 **Generating supportive files**
 
 
When doing predictions on a set, we always save the results in 3 .npy files -  `predictions.npy`, `image_indices.npy`, `patch_labels.npy`.

* `predictions.npy` is a list for all bags with their instance predictions. Instance predictions are multi dimensional array of 16x16x1 (3rd dimension is the class dimension, but we predict 1 class).
* `patch_labels.npy` contains the corresponding ground-truth instance labels for each bag. Bags with no segmentation are assigned labels of only 0s or 1s on all their patches, depending on the bag label. For example, a positive bag is assigned have 1 for all its instance labels. 
* `image_indices.npy` contains the image index(name) of each bag, in the same order as corresponding to each index.  


 `evaluate_performance.py` - the script works in 2 steps, where it generates a few more supportive files and finally computes the performance of the model. The generated files are:

   * `accurate_localization.npy` saves the accurate instance localizations.  
  Firstly, the instance predictions of each bag are binarized, using a threshold of 0.5, such that active patches have a label '1', and inactive of '0'. The active patches of each bag are compared to the instance labels when available. If the intersection over union (IoU) between the two is greater than 0.1, the instance localization is considered as accurate. 
 * `dice.npy` saves the dice score for every annotated image.

  * `bbox_present.npy` keeps memory of the images with present or absent annotation.
    
    `accurate_localization`, `dice` and `bbox_present` are always generated. For datasets without annotated images, these files are simply futile. 
 
 * `image_labels.npy` contains the ground-truth label of each bag. The file is generated from  `patch_labels.npy`, where at least one active patch leads to a positive label on bag level, and else the bag is negative.
* `image_predictions.npy` keeps the prediction for each image on bag level.

  What is more, the implementation allows different bag level predictions accroding to the pooling
operator - `LSE, NOR, Mean`

  **Evaluating performance**

  After generating all of the aforementioned file, the script compares the predictions to the ground-truth labels on instance and bag level. On instance level, the accuracy from IoU and DICE score is computed, and on bag level the AUC are computed and saved in .csv files. Additionally, confusion matrix and roc curve visualization are generated and saved.
  Finally, the script is developed for a single class prediction.  

#### Stability
`stability` module is comprised also of several sub-modules. `script` modules contains all scripts, which perform experiments with the stability and the proposed score.

`simulate_score_behavior.npy` runs experiments of how the proposed scores behave with various proportions of agreeing and disagreeing predictions.

 `do_stability_all_classifiers.npy` runs all the experiments for stability. 
   * `stability_all_classifiers()` saves .csv files of stability for each image across classifiers,
            visualizations for each stability score across classifiers and per image,
             visualizations of nan values of the stability scores
   * `stability_all_classifiers_instance_level()` investigates the instance performance against the stability 


## Configuration
 In order to run any script, the settings are taken from a global configuration file. `examples_config.yml` shows the
  structure of the file, and the values expected.

While some of the parameters are clear in their function and role, others are more obfuscating. Here we describe the
role and effect of each parameter:

* `skip_processing_labels`:
If false, then the raw data and csv files are read, preprocessed and used. This is computationally heavier. So after reading raw labels, a preprocessed csv is created, which can be used instead for next times. \
If true, preprocessing is skipped and the generated csv is read directly from the specified path.
* `train_mode`:
If true, then some scripts train a neural network from the training set \
If false, same scripts instead save predictions on train, validation, and test set

* `use_xray_dataset`:
If true, xray dataset is used and paths to xray files \
* `class_name`: The class used for training and prediction. Xray classes are typed with first capital letter, and MURA classes are typed lowercase.   (ex: "Cardiomegaly", 'shoulder')
* `mura_interpolation`:
If true, interpolation method is used for resizing images. If false, padding. For xray, `interpolation=true`, else `interpolation=false`.
* `pascal_image_path`: path to pascal images
* `use_pascal_dataset`: true/false - **If false and `use_xray_dataset`=false, then MURA is used**

* `image_path`: path to xray images
* `classication_labels_path`: path to chest XRay Data_Entry_2017.csv
* `localization_labels_path`: path to chest XRay Bbox_List_2017.csv
* `results_path`: folder where general results are stored
* `processed_labels_path`: path to preprocessed csv of xray dataset

* `mura_train_img_path`: path to MURA training images (as defined according to the authors)   
* `mura_train_labels_path`: path to labels of training images
* `mura_test_img_path`:  path to MURA validation images (original split, from the authors), but we use them for testing   
* `mura_test_labels_path`: path to labels of validation (original split, from the authors) images
* `mura_processed_train_labels_path`: path to preprocessed train csv of mura dataset
* `mura_processed_test_labels_path`: path to preprocessed test csv of mura dataset

All of the next paths are locations for saving results.
* `results_path`: path to save general results, e.g. statistics during training process
* `prediction_results_path`: path to save .npy files of the predictions generated
* `trained_models_path`: path to save trained models
* `stability_results`: path to results from stability experiments
