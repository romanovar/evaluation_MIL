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
label_df = ld.get_classification_labels(label_dir="C:/Users/s161590/Desktop/Data/X_Ray/test.csv")
# loads Y as a list
# X, Y = ld.load_process_png(label_df)

X, Y = ld.load_process_png_v2(label_df)

# Y = ld.drop_extra_label_columns(Y)
X_train, X_test, Y_train, Y_test = ld.split_test_train(X, Y)
# X_train, X_test = ld.split_test_train(X)


# X_train, X_test, Y_train, Y_test = X_tr.to_numpy(), X_t.to_numpy(), Y_tr.to_numpy(), Y_t.to_numpy()
# print(Y.shape)
print(type(X_train))
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


train_acc, test_acc, parameters = model.model(X_train, Y_train, X_test, Y_test, P=16, start_learning_rate = 0.001,
          num_iter = 5, num_epochs = 10, minibatch_size = 5, print_cost = True, plt=None)