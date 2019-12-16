from sklearn.metrics import roc_auc_score


def compute_auc_1class(labels_all_classes, img_predictions_all_classes):
    auc_score = roc_auc_score(labels_all_classes[:, ], img_predictions_all_classes[:, ])
    return auc_score


def get_image_index_xray(image_link_collection, image_ind):
    return image_link_collection[image_ind][-16:-4]


def get_image_index_mura(image_link_collection, image_index):
    patient_id_image_id = (image_link_collection[image_index].partition("/patient"))[-1]
    # '1_positive/image1.png' to image1.png
    image_id = patient_id_image_id.partition("/study")[-1].partition("/")[-1]
    patient_id = patient_id_image_id.partition("/study")[0]
    return patient_id+'_'+image_id


def get_image_index(xray, image_link_collection, image_index):
    if xray:
        return get_image_index_xray(image_link_collection, image_index)
    else:
        return get_image_index_mura(image_link_collection, image_index)