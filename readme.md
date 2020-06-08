[This version](https://github.com/romanovar/evaluation_MIL/tree/v1.0) is the state of the code at the time of my defense.

# Overview
This repository constitutes my master graduation project. The repository includes Keras/Tensorflow implementation of
"Thoracic Disease Identification and Localization with Limited Supervision" https://arxiv.org/abs/1711.06373
for a single class (in cnn module)
and further experiments on evaluating stability of MIL algorithms (in stability module).

<!-- # Reproducibility
[More on reproducibility and project structure](reproducibility_project.md)

This is a short introduction into the project structure and reproducibility.

**Contents**
* Getting started
* Project structure

Introduction structure
 To reproduce the results of the project 3 important aspects are important:
  * virtual environment
  * script parameters
  * project modules

We will look into each one of them in details.
-->
## Getting started - Requirements
Requirements for this projects are:
- [x] Python version 3.7.x
- [x] Requirements.txt lists all the necessary libraries needed for the project and their versions.
[More about virtual environments and how to install packages from requirements.txt](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
- [x] CUDA version 10.2.89
- [x] Ubuntu 18.04.4
- [x] Nvidia driver version 440.82
- [x] GPU GeForce GTX 1080 Ti

## Datasets
We use 3 data sets, two of which are medical, and one non-medical:
 * [chest X-ray dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) [1], for short referred as X-Ray in the parameters file.
 * [MURA dataset](https://arxiv.org/abs/1712.06957) [2], for short referred as MURA
 * [Pascal VOC 2005](http://host.robots.ox.ac.uk/pascal/VOC/voc2005/index.html), for short referred as Pascal

[1]: Wang, Xiaosong et al. “ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017).

[2]: Rajpurkar, Pranav et al. “MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs.” ArXiv abs/1712.06957 (2017).


## Modules
Project is divided into 2 modules - `cnn` and `stability`.

#### CNN
`cnn` modules contains all scripts, which are in some way related to the neural architecture - this includes training of a model, evaluating its performance, preprocessing of input.
The following scripts can be ran:
* `train_model.py` allows explorative training of models.
    <details>
    <summary>Click to see required input:</summary> <br>
      In order to train the generator expects values for the input (images) and output (their labels) of the network.
      The input is a list directory paths where an input image resides. The required image shape is (512x512x3).
      The image is read, optionally preprocessed, and passed to the neural network. The labels for an image has a shape of (16, 16, 1). The first two dimensions are the patch sizes an image is divided into, and the third dimension is the number of prediction classes.  
    </details>

    <details>
    <summary>Click to see output files:</summary> <br>

    * `train_info_<IDENTIFIER>.npy` keeps loss and other settings during training for each epoch

    * `predictions_<IDENTIFIER>.npy` are the raw predictions for a set. It is a list for all bags with their instance predictions. Instance predictions are multi dimensional array of 16x16x1 (3rd dimension is the class dimension, but we predict 1 class).

    * `patch_labels_<IDENTIFIER>.npy` contains the corresponding ground-truth instance labels for each bag. Bags with no segmentation are assigned labels of only 0s or 1s on all their patches, depending on the bag label. For example, a positive bag is assigned have 1 for all its instance labels.

    * `image_indices_<IDENTIFIER>.npy` are the sample unique identifiers in a set. It contains the image index(name) of each bag, in the same order as corresponding to each index.  
     </details>
  
  More on data preparation in [Data preparation section](#data-preparation).

* `run_cross_validation.py` performs cross validation (CV) on a specific architecture. In xray, every split is yields different train/validation/testing set. In Mura, the testing set is fixed, it is equal to the validation set from the author split.

     <details>
      <summary>Click to see output files:</summary>

      The same as in `train_model.npy`. `<IDENTIFIER> = <SAMPLE_SET>_CV<CV_fold>`.

    </details>

* `train_models_on_subset.py` trains several models on similar training subset. Currently it trains 5 models on 95% of the original training set. Initially, the script takes a specific cross validation split of training, validation and testing set, and then drops a portion of the samples from the training set. The validation and testing set are preserved the same among all models. Later in Stability module we test the stability of the performance of these 5 models trained on highly similar data.

    <details>
   <summary>Click to see output files:</summary> <br>    

      The same as in `train_model.npy`. ``<IDENTIFIER> = <SAMPLE_SET>_CV<CV_fold>``.

   </details>

* `evaluate_performance.py` evaluates the performance of a trained model. It calculates AUC for the set. If segmentation labels are available - it is calculated the dice coefficient and accuracy from IOU (with threshold of 0.1).

    <details>
     <summary>Click to see output files:</summary> <br>    

     * `image_labels_<IDENTIFIER>.npy` contains the ground-truth label of each bag. The file is generated from `patch_labels_<IDENTIFIER>.npy`, where at least one active patch leads to a positive label on bag level, and else the bag is negative.

     * `image_predictions_<IDENTIFIER>.npy` keeps the image level prediction for each image. The value is aggregated from the raw predictions in `predictions_<IDENTIFIER>.npy`.

     * `bbox_present_<IDENTIFIER>.npy` keeps memory of the images with present or absent annotation. A list with 0/1 flag for each image.

     * `dice_<IDENTIFIER>.npy` saves the dice score for every annotated image. If segmentation is not available, then the image is assigned an invalid dice score of -1.

     ``<SAMPLE_SET>`` denotes a specific set - e.g. train, validation or test set <br>
        ``<CV_fold>`` denotes a specific cross validation fold number
     </details>

     <details>
      <summary>Click to see input files:</summary> <br>    
      The script expects the output files from any of the training scripts:
      `predictions_<IDENTIFIER>.npy`, `patch_labels_<IDENTIFIER>.npy` and `image_indices_<IDENTIFIER>.npy`. The user has to specify within the script  the value of `IDENTIFIER`.

    </details>

    <details>
       <summary>Click to see additional output files:</summary> <br>    
       Additional output files are output files that saves the generated results, but are not necessary for any of the following scripts. Here this script generates the following scripts:

   * `dice_inst_<IDENTIFIER>.csv` - saves the dice score per image for all samples with available annotation
   - `evaluation_performance_<IDENTIFIER>.csv` saves a .csv file a column for each of the evaluation metrics used.  'accuracy' column saves the average accuracy for all segmentation images in the set. Accuracy is calculated from IOU with threshold 0.1. 'dice' columns saves the average dice score for all segmentation images in the set. AUC column saves the AUC score for the all images in the set.  

   Visualizations generated:
   - `roc_curve_<IDENTIFIER>.jpg` - ROC curve based on the Area Under the Curve metric
   - `confusion_matrix_<IDENTIFIER>.jpg` confusion matrix of the predictions (with the actual number of samples per group )
   - `confusion_matrix_<IDENTIFIER>_norm.jpg`  confusion matrix of the predictions represented as normalized value from the whole true label group

   </details><br>

* `preprocess_images.py` This is an *optional* script. It preprocess the input images to the format required during training. Preprocessed images are saved in a new directory (requiring more memory), and during training the saved preprocessed images are directly fed into the neural network. Thus, the training procedure is quicker. The script does not preprocess all images from a dataset, but only the one that are used and necessary. So changing the prediction class may require running this script again. If the images are not preprocessed in advance, the preprocessing step is incorporated within the training generator. That, however, slows the training procedure.
    **Currently this script is available only for the Xray dataset.**     



#### Stability
`stability` module is comprised also of several sub-modules. `script` modules contains all scripts, which perform experiments with the stability and the proposed score.

* `simulate_stability_score.npy` This is an *optional* runs experiments of how the proposed scores behave with various proportions of agreeing and disagreeing predictions. This script is used to show certain behavior of several viable alternatives for stability score. It is not needed for the computation of stability.

* `run_stability.npy` runs all the experiments for stability. saves .csv files of stability for each image across classifiers, creates visualizations for each stability score across classifiers and per image, visualizations of nan values of the stability scores. It also investigates the instance performance against the stability.

  <details>
   <summary>Click to see input files:</summary>     
   All input and output files from `evaluate_performance.py`
   </details>
   <details>
    <summary>Click to see output files:</summary>

    * `mean_stability_bbox.csv` saves the stability score of  all images with available segmentation. The table shows the mean values of several stability scores (mean positive Jaccard, mean corrected positive Jaccard, mean corrected IOU, mean Spearman) for each image, together with the mean dice score. The mean values are calculated by aggregating the stability/dice score across all models for the same image. In this way we can sees some patterns between well segmented images (high avg dice score and low std dev of dice) and the stability score.

    * `mean_stability_all_img.csv` saves the stability score of all images across models. The table shows the mean values of several stability scores (mean positive Jaccard, mean corrected positive Jaccard, mean corrected IOU, mean Spearman) for each image. The stability scores are aggregated across the different models for the same image. This information can be used for further analysis e.g. revealing differences in values of the stability scores for the same image; or analyzing images with highest stability.
  </details>

    <details>
     <summary>Click to see generated visualizations :</summary>    

     - `correlation_<STABILITY_SCORE>_nan_<IMG_SUBSET>.jpg` shows a heatmap visualization of the NaN values of a specific stability score (`<STABILITY_SCORE>`) between 5 classifiers. Since the stability score is a pairwise score between two models. So the heatmap shows the NaN values from the stability scores between each two pairs of models. Depending on the user's choice, the analysis is done on all images (``<IMG_SUBSET>`` = all_img) or only on images with positive label (``<IMG_SUBSET>`` = pos_img).     

     - `correlation_<STABILITY_SCORE>_nan_norm_<IMG_SUBSET>.jpg` shows a heatmap visualization of the NaN values of a specific stability score (`<STABILITY_SCORE>`) between 5 classifiers. Since the stability score is a pairwise score between two models, so the heatmap shows the **ratio of** NaN values from the stability scores between each two pairs of models. Depending on the user's choice, the analysis is done on all images (``<IMG_SUBSET>`` = all_img) or only on images with positive label (``<IMG_SUBSET>`` = pos_img).        

  - `correlation_mean_<STABILITY_SCORE>_<IMG_SUBSET>.jpg` shows a heatmap visualization of the mean values of a specific stability score (`<STABILITY_SCORE>`) between 5 classifiers. Since the stability score is a pairwise score between two models, we compare the predictions of all images between two models and aggregate them to a mean value. Depending on the user's choice, the analysis is done on all images (``<IMG_SUBSET>`` = all_img) or only on images with positive label (``<IMG_SUBSET>`` = pos_img).     

  - `scatter_mean_<STABILITY_SCORE>_mean_dice.jpg` These visualizations are generated only for images with available segmentation. It visualizes a scatter plot for each segmentation image - where y-axis is the mean dice score for an image across all models and x-axis is the mean stability score (`<STABILITY_SCORE>`). In this way we can sees some patterns between well segmented images (high avg dice score and low std dev of dice) and the stability score.  

 </details>


## Data preparation
   Input images are expected to have a shape of (512x512x3). We pass a directory path to an image and the generator takes care that the image has the required size. Currently the following three scenarios are differentiated in the generator during training:
-  Image with size (K x K x 3) where K > 512pi. In this case the image is decreased to (512 x 512 x 3).  
- Images with size (M x N x 3) where M > 512pi and N < 512pi. In this case the image is first proportionally decreased such that its bigger axis is 512pi. The smaller axis is padded with black color to a size of 512pi.
- Images with size (P x Q x 3) where P, Q < 512pi. In this case, both sides of the image are padded with black color to a size of 512pi.

Labels are expected to have a shape of (16 x 16 x 1), because the images are divided into 16x16 patches. Third dimension encodes the number of prediction classes. Currently, we predict binary labels.
Segmentation annotation are translated as follows: Segmentations are converted to a rectangular shape. The label mask is divided into the same patch sizes as the image. Finally, each patch receives label of '1' if part of the segmentation is located on it. Else the patch is label with '0'.        
 For images with no available segmentation (only image/bag label) the patch/instance labels take the label of the image. Eg. For positive image with no segmentation the label is a matrix of (16 x 16 x 1) with 1s. For negative image, the label is a matrix of size (16 x 16 x 1) with 0s.


## Configuration File
 In order to run any script, the settings are taken from a global configuration file. `examples_config.yml` shows the
  structure of the file, and the values expected.

While some of the parameters are clear in their function and role, others are more obfuscating. Here we describe the
role and effect of each parameter:
* `resized_images_before_training`: Currently working only for X-Ray dataset. If true, the parameter triggers preprocessing to start in `preprocess_images.py`, in training scripts (`train_models.py`, `run_cross_validation.py`, `train_models_on_subsets.py`) the parameter indicates that the input images are already preprocessed and can be just loaded without any processing needed. 
* `skip_processing_labels`:
If false, then the raw data and csv files are read, preprocessed and used. This is computationally heavier. So after reading raw labels, a preprocessed csv is created, which can be used instead for next times. \
If true, preprocessing is skipped and the generated csv is read directly from the specified path.
* `train_mode`:
 The parameter is applicable only in the training scripts (`train_models.py`, `run_cross_validation.py`, `train_models_on_subsets.py`). 
* `train_mode`: The parameter is applicable only in the training scripts (`train_models.py`, `run_cross_validation.py`, `train_models_on_subsets.py`). 
If true, the parameter triggers training procedure in the training scripts \
If false, the script loads a preciously trained and saved model and does predictions on the train, validation, and test set.

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
