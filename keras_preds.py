import numpy as np
import load_data as ld
import yaml
import argparse
import keras_utils
import os
import tensorflow as tf
from custom_accuracy import keras_accuracy, compute_image_probability_asloss, combine_predictions_each_batch, \
    make_save_predictions, compute_auc, list_localization_accuracy, compute_image_probability_production
from custom_loss import keras_loss, test_compute_ground_truth_per_class_numpy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

skip_processing = config['skip_processing_labels']
image_path = config['image_path']
classication_labels_path = config['classication_labels_path']
localization_labels_path = config['localization_labels_path']
results_path = config['results_path']
processed_labels_path = config['processed_labels_path']
train_mode = config['train_mode']
test_single_image = config['test_single_image']
prediction_skip_processing = config['prediction_skip_processing']

#######################################################################
def load_npy(file_name, res_path):
    return np.load(res_path + '/' + file_name, allow_pickle=True)


def get_index_label_prediction(file_set_name, res_path):
    prediction_file = 'predictions_' + file_set_name + '.npy'
    img_ind_file = 'image_indices_' + file_set_name + '.npy'
    patch_labels_file = 'patch_labels_' + file_set_name + '.npy'

    preds = load_npy(prediction_file, res_path)

    img_indices = load_npy(img_ind_file, res_path)
    patch_labs = load_npy(patch_labels_file, res_path)

    print(preds[1, :, :, 0])
    print(img_indices)
    return preds, img_indices, patch_labs


def get_label_prediction_image_level(preds, patch_labs, img_pred_as_loss):
    if img_pred_as_loss== 'as_loss':
        img_labels, img_prob_preds_v1 = compute_image_probability_asloss(preds, patch_labs, P=16)
    elif img_pred_as_loss== 'as_production':
        img_labels, img_prob_preds_v1 = compute_image_probability_production(preds, patch_labs, P=16)

    # _, img_prob_preds_v2 = compute_image_probability_production(preds, patch_labs, P=16)
    # _, img_prob_preds_v3 = compute_image_probability_production_v2(preds, patch_labs, P=16)
    # return img_labels, img_prob_preds_v1, img_prob_preds_v2, img_prob_preds_v3
    return img_labels, img_prob_preds_v1


# def evaluate_tensors(img_labels, img_preds_v1, img_preds_v2, img_preds_v3, acc_per_class):
#     with sess.as_default():
#         image_labels, preds_v1, preds_v2, preds_v3 = img_labels.eval(), img_preds_v1.eval(), img_preds_v2.eval(), \
#                                                      img_preds_v3.eval()
#         acc_all_class = acc_per_class.eval()
#     return image_labels, preds_v1, preds_v2, preds_v3, acc_all_class
#


def process_prediction_per_batch(predictions, patch_labels, img_pred_as_loss, l_bound, r_bound, coll_image_labs,
                                 coll_image_preds, coll_accurate_preds, coll_bbox_present, k):
    batch_pred = predictions[l_bound:r_bound, :, :, :]
    batch_patch_lab = patch_labels[l_bound:r_bound, :, :, :]

    #TODO: Different ways of predicting image label
    batch_img_labels, batch_img_preds_v1 = get_label_prediction_image_level(batch_pred, batch_patch_lab, img_pred_as_loss)
    # elif img_pred_as_loss=='as_production':
    #     batch_img_labels, batch_img_preds_v1 = get_label_prediction_image_level(batch_pred, batch_patch_lab)

    acc_loc, acc_preds, nr_bbox_present = list_localization_accuracy(batch_patch_lab, batch_pred)

    with tf.Session().as_default():
        batch_image_labels, batch_image_predictions_v1 = batch_img_labels.eval(), batch_img_preds_v1.eval()

        coll_image_labs = combine_predictions_each_batch(batch_image_labels, coll_image_labs, k)
        coll_image_preds = combine_predictions_each_batch(batch_image_predictions_v1, coll_image_preds, k)
        print(batch_image_labels.shape)

    # acc_loc= tf.Session().run(acc_loc)
    acc_predictions = np.asarray(tf.Session().run(acc_preds))
    acc_predictions = np.expand_dims(acc_predictions, axis=0)

    total_bbox_present = np.asarray(tf.Session().run(nr_bbox_present))
    total_bbox_present = np.expand_dims(total_bbox_present, axis=0)
    print("acc_loc, acc_preds, nr_bbox_present")
    print(k)

    coll_accurate_preds = combine_predictions_each_batch(acc_predictions, coll_accurate_preds, k)
    coll_bbox_present = combine_predictions_each_batch(total_bbox_present, coll_bbox_present, k)
    # print(coll_accurate_preds.shape)

    return coll_image_labs, coll_image_preds, coll_accurate_preds, coll_bbox_present


def process_prediction_all_batches(predictions, patch_labels, img_pred_as_loss, batch_size, file_unique_name):
    # m = predictions.shape[0]  # number of training examples
    # batch_size = np.math.ceil(predictions.shape[0] / total_batch_nr)

    num_complete_minibatches = (np.floor(predictions.shape[0] / batch_size)).astype(int)

    coll_image_labels = 0
    coll_image_predictions = 0
    coll_accurate_preds = 0
    coll_bbox_present = 0

    for k in range(0, num_complete_minibatches):
        l_bound = k * batch_size
        r_bound = (k + 1) * batch_size
        coll_image_labels, coll_image_predictions, coll_accurate_preds, coll_bbox_present = process_prediction_per_batch(
            predictions,
            patch_labels, img_pred_as_loss, l_bound,
            r_bound, coll_image_labels,
            coll_image_predictions,
            coll_accurate_preds, coll_bbox_present,
            k)

    if predictions.shape[0] % batch_size != 0:
        l_bound = (num_complete_minibatches) * batch_size
        r_bound = predictions.shape[0]
        # ratio_last_batch = (predictions[l_bound:r_bound, :, :, :].shape[0] / batch_size)
        coll_image_labels, coll_image_predictions, coll_accurate_preds, coll_bbox_present = process_prediction_per_batch(
            predictions,
            patch_labels, img_pred_as_loss, l_bound,
            r_bound, coll_image_labels,
            coll_image_predictions,
            coll_accurate_preds,
            coll_bbox_present,
            k=1)

    np.save(results_path + '/image_labels_' + file_unique_name + '_'+img_pred_as_loss, coll_image_labels)
    np.save(results_path + '/image_predictions_' + file_unique_name + '_'+img_pred_as_loss , coll_image_predictions)
    np.save(results_path + '/accurate_localization_' + file_unique_name, coll_accurate_preds)
    np.save(results_path + '/bbox_present_' + file_unique_name, coll_bbox_present)
    return coll_image_labels, coll_image_predictions, coll_accurate_preds, coll_bbox_present

    # batch_pred  = predictions[batch_size * total_batch_nr: predictions.shape[0], :, :, :]
    # batch_patch_lab  = patch_labels[batch_size * total_batch_nr: predictions.shape[0], :, :, :]
    # batch_img_labels, batch_img_preds_v1 = get_label_prediction_image_level(batch_pred, batch_patch_lab)
    #
    # with tf.Session().as_default():
    #     batch_image_labels, batch_image_predictions_v1 = batch_img_labels.eval(), batch_img_preds_v1.eval()
    #
    #     coll_image_labels = combine_predictions_each_batch(batch_image_labels, coll_image_labels, k)
    #     coll_image_predictions = combine_predictions_each_batch(batch_image_predictions_v1,
    #                                                             coll_image_predictions, k)
    #
    # mini_batches.append(mini_batch)


def process_prediction(file_unique_name, res_path, img_pred_as_loss, batch_size):
    # file_unique_name = 'test_set'
    predictions, image_indices, patch_labels = get_index_label_prediction(file_unique_name, res_path)

    # batch_size = np.math.floor(predictions.shape[0] / total_batch_nr)
    # for idx in range(0, total_batch_nr):
    #     l_bound = idx * batch_size
    #     r_bound = (idx + 1) * batch_size
    #     pred_batch = predictions[l_bound:r_bound, :, :, :]
    #
    # acc_per_class = accuracy_bbox_IOU(predictions, pred_batch, P=16, iou_threshold=0.1)

    # img_labels_v1, img_preds_v1 = get_label_prediction_image_level(predictions, patch_labels)

    # image_labels, preds_v1, preds_v2, preds_v3, acc_all_class = evaluate_tensors(img_labels_v1, img_preds_v1,
    #                                                                              img_preds_v2, img_preds_v3, acc_per_class)

    ##TODO: currently saving image labels and image predictions
    coll_image_labels, coll_image_predictions, coll_accurate_preds, coll_bbox = process_prediction_all_batches(
        predictions,
        patch_labels, img_pred_as_loss, batch_size,
        file_unique_name)

    ## TODO: compute accuracy per bacth and consider the size of the last batch - DONE AT THE END
    accurate_pred_all_batches = np.sum(coll_accurate_preds, axis=0)

    total_bbox_all_batches = np.sum(coll_bbox, axis=0)
    print("accurate pred all batches")
    print(accurate_pred_all_batches)
    print(total_bbox_all_batches)
    ## TODO: computing auc - just loading image labels and image predicitons from npy - AT THE END OF THE SESSION
    # with tf.Session().as_default():
    #     auc_all_classes_v1 = compute_auc(img_labels_v1.eval(), img_preds_v1.eval())
    #     keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v1, 'auc_prob_' + file_unique_name + '_v1.csv',
    #                                         results_path)
    # auc_all_classes_v2 = compute_auc(image_labels, preds_v2)
    # auc_all_classes_v3 = compute_auc(image_labels, preds_v3)

    # keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v2, 'auc_prob_'+ file_unique_name+ '_v2.csv', results_path)
    # keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v3, 'auc_prob_'+ file_unique_name+ '_v3.csv', results_path)
    # with tf.Session().as_default():
    # with tf.Session().as_default():
    #     acc_all_class = acc_per_class.eval()
    #     keras_utils.save_accuracy_results(ld.FINDINGS, acc_all_class, 'accuracy.csv', results_path)
    # # return auc_all_classes_v1, auc_all_classes_v2, auc_all_classes_v3, acc_all_class
    # return auc_all_classes_v1, acc_all_class


def load_img_pred_labels(file_set, img_pred_as_loss, res_path):
    img_labs_file = 'image_labels_' + file_set + '_'+img_pred_as_loss+'.npy'
    img_preds_file = 'image_predictions_' + file_set + '_'+img_pred_as_loss+'.npy'

    img_preds = load_npy(img_labs_file, res_path)
    img_labels = load_npy(img_preds_file, res_path)
    img_labels = img_labels.astype(int)
    return img_labels, img_preds


def load_accuracy_localization(file_set, res_path):
    acc_local = 'accurate_localization_' + file_set + '.npy'
    bbox_present = 'bbox_present_' + file_set + '.npy'

    acc_local = load_npy(acc_local, res_path)
    bbox_present = load_npy(bbox_present, res_path)
    # img_labels = img_labels.astype(int)
    return acc_local, bbox_present


def compute_save_accuracy(acc_loc, bbox_pres, file_unq_name):
    sum_acc_loc = np.sum(acc_loc, axis=0)
    sum_bbox_pres = np.sum(bbox_pres, axis=0)
    sum_acc_local_all = np.sum(acc_loc)
    sum_bbox_present_all = np.sum(bbox_pres)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_class = sum_acc_loc / sum_bbox_pres
        acc_avg = sum_acc_local_all/sum_bbox_present_all
    local_col_names = [ld.FINDINGS[i] for i in [0, 1, 4, 8, 9, 10, 12, 13]]
    keras_utils.save_evaluation_results(local_col_names, acc_class, 'accuracy_' + file_unq_name + '.csv',
                                        results_path, add_col='Avg_accuracy', add_value=acc_avg)


def do_predictions_set(data_set_name, skip_pred_process, img_pred_as_loss):
    if not skip_pred_process:
        process_prediction(data_set_name, results_path, img_pred_as_loss, batch_size=2)

    img_labels, img_preds = load_img_pred_labels(data_set_name, img_pred_as_loss, results_path)
    acc_local, bbox_present = load_accuracy_localization(data_set_name, results_path)

    compute_save_accuracy(acc_local, bbox_present, data_set_name)
    auc_all_classes_v1 = compute_auc(img_labels, img_preds)
    keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v1, 'auc_prob_' + data_set_name + '_v1.csv',
                                        results_path)
#########################################################

dataset_name = 'test_set'
image_prediction_method = 'as_loss'
do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)


dataset_name = 'train_set'
image_prediction_method = 'as_loss'
do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)

dataset_name = 'val_set'
image_prediction_method = 'as_loss'
do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)


# sum_acc_local =np.sum(acc_local, axis=0)
# sum_bbox_present = np.sum(bbox_present, axis=0)
# print("sums")
# print(sum_acc_local)
# print(sum_bbox_present)
# # out = ["missing value" for x in range(0,9)]
#
# with np.errstate(divide='ignore',invalid='ignore'):\
#     acc = (sum_acc_local/sum_bbox_present)
# # acc2 = np.divide(sum_acc_local, sum_bbox_present, out=out,  where=sum_bbox_present!=0)
# print(acc)
# local_col_names = [ld.FINDINGS[i] for i in [0, 1,4, 8, 9, 10, 12,13 ]]
# local_col_names.append('Average')
# A = [0.5]
# res_con = np.concatenate( ( acc,  A[np.newaxis,:]), axis=1)
# print(local_col_names)
# print(acc)
# print(res_con)


#########################################################


# file_unique_name = 'test_set'
#
# prediction_file = 'predictions_'+file_unique_name+'.npy'
# img_ind_file = 'image_indices_'+ file_unique_name+ '.npy'
# patch_labels_file = 'patch_labels_'+ file_unique_name+ '.npy'
#
# predictions = np.load(results_path + '/'+prediction_file)
# image_indices = np.load(results_path + '/'+img_ind_file)
# patch_labels = np.load(results_path+'/'+patch_labels_file)
# #
# # print(predictions)
# # print("**************")
# # # print(predictions[3])
# # print("**************")
# # print(image_indices)
# # print(patch_labels[3, :, :,1 ])
# # print(predictions[3, :, :, 1])


# img_labels_v1, img_prob_preds_v1 = compute_image_probability_asloss(predictions, patch_labels, P=16)
# img_labels_v2, img_prob_preds_v2 = compute_image_probability_production(predictions, patch_labels, P=16)
# img_labels_v3, img_prob_preds_v3 = compute_image_probability_production_v2(predictions, patch_labels, P=16)

##### ACCURACY ###############

# sess = tf.Session()
# with sess.as_default():
#     # img_lab_v1, preds_v1 = img_labels_v1.eval(), img_prob_preds_v1.eval()
#     # img_lab_v2, preds_v2 = img_labels_v2.eval(), img_prob_preds_v2.eval()
#     # img_lab_v3, preds_v3 = img_labels_v3.eval(), img_prob_preds_v3.eval()
#     image_labels, preds_v1, preds_v2, preds_v3 = img_labels_v1.eval(), img_preds_v1.eval(), img_preds_v2.eval(),\
#                                                  img_preds_v3.eval()
#     acc_all_class = acc_per_class_V2.eval()
#     print(acc_all_class)
#



##### AUC ####################
# TO FIX RANGE compute auc
# auc_all_classes_v1 = compute_auc(image_labels, preds_v1)
# auc_all_classes_v2 = compute_auc(image_labels, preds_v2)
# auc_all_classes_v3 = compute_auc(image_labels, preds_v3)

# keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v1, 'auc_prob_'+ file_unique_name+ '_v1.csv', results_path)
# keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v2, 'auc_prob_'+ file_unique_name+ '_v2.csv', results_path)
# keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v3, 'auc_prob_'+ file_unique_name+ '_v3.csv', results_path)
# keras_utils.save_accuracy_results(ld.FINDINGS, acc_all_class, 'accuracy.csv', results_path)



############################# TEST SET ########################

# auc_v1_test, auc_v2_test, auc_v3_test, acc_test = process_prediction('test_set', results_path)

# auc_v1_val, auc_v2_val, auc_v3_val, acc_val = process_prediction('val_set', results_path)

# auc_v1_train, auc_v2_train, auc_v3_train, acc_train = process_prediction('train_set', results_path)



# keras_utils.plot_grouped_bar_auc(auc_v1_train, auc_v1_val, auc_v1_test, 'auc_v1', results_path, ld.FINDINGS)
# keras_utils.plot_grouped_bar_auc(auc_v2_train, auc_v2_val, auc_v2_test, 'auc_v2', results_path, ld.FINDINGS)
# keras_utils.plot_grouped_bar_auc(auc_v3_train, auc_v3_val, auc_v3_test, 'auc_v3', results_path, ld.FINDINGS)
#
# keras_utils.plot_grouped_bar_accuracy(acc_train, acc_val, acc_test, 'accuracy', results_path, ld.FINDINGS)
