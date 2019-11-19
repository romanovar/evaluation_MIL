from sklearn.metrics import roc_auc_score


def compute_auc_1class(labels_all_classes, img_predictions_all_classes):
    auc_score = roc_auc_score(labels_all_classes[:, ], img_predictions_all_classes[:, ])
    return auc_score