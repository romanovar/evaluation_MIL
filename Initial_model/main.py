import numpy as np
import readDCMfiles as rdf
import readNiifiles as rnf
import tensorflow as tf
import CNN


def get_sorted_ann_list(z_dict):
    z_list = []
    for key in sorted(z_dict, reverse=True):
        z_list.append(z_dict[key])
    return np.reshape(z_list, newshape=(len(z_list), 1))


test_dir= "C:/Users/s161590/Desktop/Data/new/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
image, z_coordinates = rdf.load_dcm_data(test_dir)
print(image.shape)

z_labels = rnf.load_annotation_files(z_coordinates)
z_array = get_sorted_ann_list(z_labels)
print(z_array.shape)

# images_train, images_test, y_train, y_test  = train_test_split(image, z_list, test_size=0.20, random_state=3)
#60% training set, 20% dev set, 20% test set

X_train, X_test, Y_train, Y_test = image, image, z_array, z_array
# indices=[0,1]
# depthk=1
# print(tf.one_hot(indices, depthk))

Y_test_enc = tf.one_hot(Y_test, depth=1, on_value=1, off_value=0)
Y_train_enc = tf.one_hot(Y_train, depth=1, on_value=1, off_value=0)

(m, n_H0, n_W0, n_C0) = X_train.shape
X, Y = CNN.create_placeholders(n_H0,n_W0,n_C0, Y_train.shape[1])
print(X, Y)
# Y_train_enc = to_categorical(Y_tr, dtype=float)


