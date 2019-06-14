import tensorflow as tf
import model
import load_data as ld
import numpy as np
#
# #############################################
# ######## CODE FOR bounding boxes ############
# #############################################
# bbox = load_csv("C:/Users/s161590/Desktop/Data/X_Ray/BBox_List_2017.csv")
# # ann_list = get_ann_list(bbox)
# # x = fe.get_process_annotated_png(ann_list)
# # features_x = fe.get_feature_extraction(x)
#
#############################################
######## CODE FOR classification ############
#############################################
label_df = ld.get_classification_labels()
print(label_df.shape)
X, Y = ld.load_process_png(label_df)

x_fe = tf.placeholder(tf.float32, shape=(None, 16, 16, 2048))
features_x = model.get_feature_extraction(X)
MP = model.divide_image_to_patches(x_fe)
pred = model.recognition_network(MP)


# # x_20x20 = MaxPooling2D(pool_size=13, strides=1, padding="valid")
# # x_16x16 = MaxPooling2D(pool_size=17, strides=1, padding="valid")
# # x_12x12 = MaxPooling2D(pool_size=21, strides=1, padding="valid")


def compute_image_label_classification(nn_output):
    subtracted_prob = 1 - nn_output
    flat_mat = tf.reshape(subtracted_prob, (-1, 2 * 2, 3))

    ## Normalization between [a, b]
    ### ( (b-a) (X - MIN(x)))/ (MAX(x) - Min(x)) + a
    normalized_mat = ((1 - 0.98) * (flat_mat - tf.reduce_min(flat_mat, axis=1)) /
                      (tf.reduce_max(flat_mat, axis=1) - tf.reduce_min(flat_mat, axis=1))) + 0.98

    ### KEEP the dimension for observations and for the classes
    # flat_mat = tf.reshape(normalized_mat, (-1, 2 * 2, 3))
    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return (tf.cast(1, tf.float16) - element_product)


def compute_image_label_localization(nn_output, y_true):
    pos_patches = tf.reshape((nn_output * y_true), (-1, 2 * 2, 3))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, 2 * 2, 3))

    normalized_pos = ((1 - 0.98) * (pos_patches - tf.reduce_min(pos_patches, axis=1)) /
                      (tf.reduce_max(pos_patches, axis=1) - tf.reduce_min(pos_patches, axis=1))) + 0.98
    normalized_neg = ((1 - 0.98) * (neg_patches - tf.reduce_min(neg_patches, axis=1)) /
                      (tf.reduce_max(neg_patches, axis=1) - tf.reduce_min(neg_patches, axis=1))) + 0.98
    print(normalized_neg)

    # TODO: to test the whole function
    Pi_pos_patches = tf.reduce_prod(normalized_pos, axis=1)
    Pi_neg_patches = tf.reduce_prod(normalized_neg, axis=1)
    return Pi_neg_patches*Pi_pos_patches


#TODO: to add localization loss function and conditions for triggering each
def compute_loss_classification(nn_output, y_true):
    epsilon = tf.pow(10, -7)

    y_hat = compute_image_label_classification(nn_output)
    ## IF image does not have bounding box
    loss_classification = - (y_hat * (tf.log(y_true+epsilon))) - ((1-y_hat)(tf.log(1-y_true+epsilon)))

    return loss_classification

#
# def compute_cost_per_class(pred, classnr, Y):
#     Y_hat  = compute_probab_classification(pred, classnr)
#     print(Y_hat.shape)
#     Y = Y[:, classnr]
#     print(Y.shape)
#     cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat, name=None))
#     # print(cost)
#     return cost

np.random.seed(0)
test = np.random.uniform(size = (1, 2, 2, 14))
test_l = np.random.randint(low=0, high=2, size=(1, 14))
# print(test)
# print(test_l)
print(compute_image_label_classification(test, 0))
