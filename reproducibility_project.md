# Reproducibility of 'Label stability'

This is a short introduction into the project structure and reproducibility.

<!-- **Contents**
* Getting started
* Project structure

Introduction structure -->
 The project can be roughly divided into 3 logical parts:
  * modules
  * script parameters
  * virtual environment

We will look into each one of them in details.

## Getting started - virtual environment
The project uses virtual environment, where requirements.txt lists all the necessary packages needed for the project.
[More about virtual environments and how to install packages from requirements.txt](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

 Once all the required packages are installed, the project can be ran.

## Script parameters
 In order to run any script, the settings are taken from a global configuration file. `examples_config.yml` shows the
  structure of the file, and the values expected.

 Important to note, we use 2 data sets:
 * [chest X-ray dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) [1], for short referred as xray in the parameters file.
 * [MURA dataset](https://arxiv.org/abs/1712.06957) [2], for short referred as mura

[1]: Wang, Xiaosong et al. “ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017).

[2]: Rajpurkar, Pranav et al. “MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs.” ArXiv abs/1712.06957 (2017).

While some of the parameters are clear in their function and role, others are more obfuscating. Here we describe the
role and effect of each parameter:

* `skip_processing_labels`:
If false, then the raw csv files is read, preprocessed and used. This is computationally heavier. So after reading raw labels, a preprocessed csv is created, which can be used instead for next times. \
If true, preprocessing is skipped and the generated csv is read directly.
* `train_mode`:
If true, then some scripts train a neural network from the training set \
If false, then some script instead of training do and save predictions on test set
* `test_single_image`: to be depricated - not currently used, leftover from old implementation
* `prediction_skip_processing`: false
* `use_xray_dataset`:
If true, it uses xray dataset and paths to xray files \
If false, it uses mura dataset and paths to mura files
* `class_name`: The class used for training and prediction. Xray classes are typed with first capital letter, and MURA classes are typed lowercase.   (ex: "Cardiomegaly", 'shoulder')
* `mura_interpolation`:
If true, interpolation method is used for images smaller than 512x512pi to achieve this size. Currently, for images bigger than 512x512, interpolation should be set to TRUE to enable proper computation. \
If false, padding is used for images smaller than 512x512pi.

* `image_path`: path to xray images
* `classication_labels_path`: path to chest XRay Data_Entry_2017.csv
* `localization_labels_path`: path to chest XRay Bbox_List_2017.csv
* `results_path`: folder where results are stored
* `processed_labels_path`: path to preprocessed csv of xray dataset

* `mura_train_img_path`: path to MURA training images (as defined according to the authors)   
* `mura_train_labels_path`: path to labels of training images
* `mura_test_img_path`:  path to MURA validation images (original split, from the authors), but we use them for testing   
* `mura_test_labels_path`: path to labels of validation (original split, from the authors) images
* `mura_processed_train_labels_path`: path to preprocessed train csv of mura dataset
* `mura_processed_test_labels_path`: path to preprocessed test csv of mura dataset

All of the next paths are locations for saving results.
* `results_path`: path to save general results, e.g. statistics during training process
* `generated_images_path`: path to save generated images  
* `prediction_results_path`: path to save .npy files of the predictions generated
* `trained_models_path`: path to save trained models
* `stability_results`: path to results from stability experiments

## Modules
Project is divided into 2 modules - `cnn` and `stability`.

#### CNN
`cnn` modules contains all scripts, which are in some way related to the neural architecture.
* `train_model.py` allows explorative training of models.

 Important to note that training and prediction processes are decoupled. So with `train_mode` parameter, the user specify if training should occur or prediction on already trained model.

* `run_cross_validation.py` performs cross validation on a specific architecture. In xray, every split is yields different train/validation/testing set. In Mura, the testing set is fixed, it is equal to the validation set from the author split.

 Important to note that training and prediction processes are decoupled. So with `train_mode` parameter, the user specify if training should occur or prediction on already trained model.

* `train_models_on_subset.py` trains models on training subset, which are later tested for stability. Initially, the script takes a split of training, validation and testing set, and then drops part of the training set.    

 Important to note that training and prediction processes are decoupled. So with `train_mode` parameter, the user specify if training should occur or prediction on already trained model.

When doing predictions on a set, we always save the results in 3 .npy files -  `predictions.npy`, `image_indices.npy`, `patch_labels.npy`.

`predictions.npy` is a list for all bags with their instance predictions. Instance predictions are multi dimensional array of 16x16x1 (3rd dimension is the class dimension, but we predict 1 class). In `patch_labels.npy` we can see the corresponding ground-truth instance labels for each bag. Bags with no segmentation are assigned labels of only 0s or 1s, depending on their bag label. For example, a positive bag is assigned instance labels with a matrix of 1s with size 16x16x1. However, only bags with available segmentation are considered for the evaluation of instances. Finally, in `image_indices.npy` we can see the image index(name) for each bag, corresponding to each index.  


* `evaluate_performance.py` - the script works in 2 steps.

 **Generating supportive files**

  The first part of the script compares the predictions to the labels, on bag and instance label, using the previously generated `patch_labels.npy` and `predictions.npy`, and generates more files used for performance evaluation.

  Firstly, the instance predictions of each bag are binarized, using a threshold of 0.5, such that active patches have a label '1', and inactive of '0'. The active patches of each bag are compared to the instance labels when available. If the intersection over union (IoU) between the two is greater than 0.1, the instance localization is considered as accurate. This information is saved in `accurate_localization.npy`. We also compute the dice score for every annotated image and the result per image is saved in `dice.npy`.

  Another generated file is `bbox_present.npy`, which has the same shape as `accurate_localization.npy`. This file, however, keeps in each element in the first dimension the total number of bags with annotated instances.

  This script generates `image_labels.npy`, as well. The file contains the ground-truth label of each bag. It has a shape **(Total instances, 1)**. The file is generated from  `patch_labels.npy`, where at least one active patch leads to a positive label on bag level, and else the bag is negative.

  `evaluate_performance.npy` computes the prediction for each image on bag level and saves it in `image_predictions.npy`. The matrix has a shape of **(Total instances, 1)**.

  What is more, the implementation allows supportive bag level predictions using the pooling

  **Evaluating performance**

  After generating all the aforementioned file, the script compares the predictions to the ground-truth labels on instance and bag level. On instance level, the accuracy from IoU and DICE score is computed, and on bag level the AUC are computed and saved in .csv files. Additionally, confusion matrix and roc curve visualization are generated and saved.
  Finally, the script is developed for a single class prediction, and evaluation of multi-classes should be taken with care.  

#### Stability
`stability` module is comprised also of several sub-modules. `script` modules contains all scripts, which perform experiments with the stability and the proposed score.

`simulate_score_behavior.npy` runs experiments of how the proposed scores behave with various proportions of agreeing and disagreeing predictions.

 `do_pairwise_stability.npy`

 `do_stability_all_classifiers.npy` runs all the experiments
