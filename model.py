from keras.applications import ResNet50V2
from keras.layers import Input
import tensorflow as tf
from tensorflow.python.ops.nn_ops import max_pool


def get_feature_extraction(x):
    input_tensor = Input(shape=(512, 512, 3))
    model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(512, 512, 3), pooling=None)
    fe_mat = model.predict(x)
    return fe_mat


def divide_image_to_patches(output_fe, P=16):
    if P==16:
        MP = max_pool(output_fe, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        return MP
    elif P==12:
        MP = max_pool(output_fe, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
        return MP


# ## PATCH SLICE of 16x16
# P1 = max_pool(x_fe, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
# # PATCH SLICE of 12x12
# P2 = max_pool(x_fe, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
#

def recognition_network(P):
    W1 = tf.get_variable("W1", [3, 3, 2048, 512], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    #todo: Warning: output number for bbox data
    W2 = tf.get_variable("W2", [1,1, 512, 14], initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    C1 = tf.nn.conv2d(P, W1, strides=[1, 1, 1, 1], padding='SAME')
    # BN1 = tf.nn.batch_normalization(C1)
    BN1 = tf.layers.batch_normalization(C1)
    A1 = tf.nn.relu(BN1)
    C2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding='SAME')
    return C2



# img_path ="C:/Users/s161590/Desktop/Data/X_Ray/images/00000001_000.png"
# img = image.load_img(img_path, target_size=(512, 512))
# x = image.img_to_array(img)
# print(x)
# x = np.expand_dims(x, axis=0)
# print(x.shape)
# x = preprocess_input(x)
# # print(x.shape)
# preds = model.predict(x)
# print(type(preds))
#
