from cnn import keras_preds


def performance_evaluation(config, dataset_name, th_binarization=0.5, th_iou=0.1):
    predict_res_path = config['prediction_results_path']
    use_xray = config['use_xray_dataset']
    class_name = config['class_name']

    image_labels, image_predictions, has_bbox, \
    accurate_localizations, dice_scores = keras_preds.process_prediction(dataset_name,
                                                                         predict_res_path,
                                                                         img_pred_as_loss='as_production',
                                                                         threshold_binarization=th_binarization,
                                                                         iou_threshold=th_iou)

    keras_preds.save_generated_files(predict_res_path, dataset_name, image_labels, image_predictions,
                                     has_bbox, accurate_localizations, dice_scores)

    if use_xray:
        keras_preds.compute_save_accuracy_results(dataset_name, predict_res_path, has_bbox, accurate_localizations)
        keras_preds.compute_save_dice_results(dataset_name, predict_res_path, has_bbox, dice_scores)
        keras_preds.compute_save_auc(dataset_name, 'as_production', predict_res_path,
                                     image_labels, image_predictions, class_name)
