# Overview
This repository constitutes my master graduation project. The repo includes Keras implementation of "Thoracic Disease Identification and Localization with Limited Supervision." https://arxiv.org/abs/1711.06373
and further experiments on evaluating MIL algorithms.

This is work in progress and most recent work can be found under development branch. 
# Content
- `load_data.py`: loads the label data sets and integrates them together
- `custom_loss.py`: computes loss function and accuracy of the predictions
- `keras_generators.py`: includes Keras data generator class
- `keras_utils.py`: includes helping functions 
- `keras_model.py`: builds the model and additional settings of training
- `keras_new_main.py`: includes the architecture and training of the network and
generating raw predictions
- `keras_preds.py`: includes the aggregation of raw results into bag level predictions
- `performance_eval.py`: is the main script for post processing of the raw predictions 


# Reproducibility 
- edit `example_config.yml` file with the corresponding configurations
- add `-c example_config.yml` as a script parameter
- run `keras_new_main.py` for preprocessing\reading csv, training the model, and
generating raw predictions 
-  run `performance_eval.py` for aggregating bag level predictions and calculating
evaluation matrices of classification and localization 
