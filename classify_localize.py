import tensorflow as tf
from tensorflow.python.ops.nn_ops import max_pool
import feature_extraction as fe
from ReadXray import load_csv, get_ann_list, get_classification_labels
import numpy as np
#
# #############################################
# ######## CODE FOR bounding boxes ############
# #############################################
# # bbox = load_csv("C:/Users/s161590/Desktop/Data/X_Ray/BBox_List_2017.csv")
# # ann_list = get_ann_list(bbox)
# # x = fe.get_process_annotated_png(ann_list)
# # features_x = fe.get_feature_extraction(x)
#
# #############################################
# ######## CODE FOR classification ############
# #############################################
# label_df = get_classification_labels()
# print(label_df.shape)
# X, Y = fe.load_process_png(label_df)
# features_x = fe.get_feature_extraction(X)
# # print(Y.shape)
# x_fe = tf.placeholder(tf.float32, shape=(None, 16, 16, 2048))
#
# ## PATCH SLICE of 16x16
# P1 = max_pool(x_fe, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
# # PATCH SLICE of 12x12
# P2 = max_pool(x_fe, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
#
# def recognition_network(P):
#     W1 = tf.get_variable("W1", [3, 3, 2048, 512], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
#     #todo: change later the output classes
#     W2 = tf.get_variable("W2", [3, 3, 512, 8], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
#
#     C1 = tf.nn.conv2d(P, W1, strides=[1, 1, 1, 1], padding='SAME')
#     # BN1 = tf.nn.batch_normalization(C1)
#     BN1 = tf.layers.batch_normalization(C1)
#     A1 = tf.nn.relu(BN1)
#     C2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding='SAME')
#     pred = tf.nn.sigmoid(C2)
#     return pred
#
# pred = recognition_network(P1)
# print(pred.shape)
#
# # x_20x20 = MaxPooling2D(pool_size=13, strides=1, padding="valid")
# # x_16x16 = MaxPooling2D(pool_size=17, strides=1, padding="valid")
# # x_12x12 = MaxPooling2D(pool_size=21, strides=1, padding="valid")


def compute_probab_classification(y_pred, classnr):
    one_mat = np.ones((y_pred.shape[0], y_pred.shape[1]))
    subtracted_prob = np.subtract( one_mat,y_pred[:,:,classnr])
    # print(y_pred[:,:, classnr])
    # print()
    # print(subtracted_prob)
    element_product = np.prod(subtracted_prob.flatten())
    # print(element_product)
    ### Prob y* given Xi
    return (1 - element_product)


def compute_cost(pred, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z3, dim=-1, name=None))
    return cost

np.random.seed(0)
test = np.random.uniform(size = (2, 2, 14))
test_l = np.random.randint(low=0, high=2, size=(1, 1,14))
print(test)
print(test_l)
print(compute_probab_classification(test, 0))