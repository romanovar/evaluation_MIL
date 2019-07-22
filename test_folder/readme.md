# Overview
This repository constitutes my master graduation project. The repo includes Keras implementation of "Thoracic Disease Identification and Localization with Limited Supervision." https://arxiv.org/abs/1711.06373
and further experiments on evaluating MIL algorithms.

# Content
- `load_data.py`: loads the label data sets and integrates them together
- `custom_loss.py`: computes loss function and accuracy of the predictions
- `keras_generators.py`: includes Keras data generator class
- `keras_utils.py`: includes helping functions 
- `keras_model.py`: builds the model and additional settings of training
- `keras_main.py`: includes the architecture and training of the network 


# Reproducibility 
- edit `example_config.yml` file with the corresponding configurations
- add `-c example_config.yml` as a script parameter
- run keras_new_main.py 
