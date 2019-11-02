from cnn import keras_preds





#########################################################
##TODO: parameterize the settigns of the function
def performance_evaluation(config, dataset_name, batch_size=10):
    predict_res_path = config['prediction_results_path']

    ################# STEP 1 ###########################
    # predict_res_path = 'C:/Users/s161590/Desktop/Project_li/single_class/5050/'

    # dataset_name = 'single_5050_train_set'
    image_prediction_method = 'as_loss'
    keras_preds.process_prediction_v2(dataset_name, predict_res_path,
                                      img_pred_as_loss=image_prediction_method,
                                      batch_size=batch_size)
    image_prediction_method2 = 'as_production'
    keras_preds.process_prediction_v2_image_level(dataset_name, predict_res_path,
                                                  img_pred_as_loss=image_prediction_method2,
                                                  batch_size=batch_size)

    ################# STEP 2 ###########################
    # dataset_name = 'single_patient_train_set'
    # image_prediction_method = 'as_loss'
    image_prediction_method2 = 'as_production'
    # predict_res_path = 'C:/Users/s161590/Desktop/Project_li/predictions/'

    keras_preds.combine_auc_accuracy_1class(dataset_name, image_prediction_method2, predict_res_path)
