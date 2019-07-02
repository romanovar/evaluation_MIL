# import tensorflow as tf
import load_data as ld
import model
import os
import numpy as np



#############################################
######## CODE FOR classification ############
#############################################
#
#
# label_df = ld.get_classification_labels("C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv", False)
# # # label_df = ld.get_classification_labels(label_dir="C:/Users/s161590/Desktop/Data/X_Ray/test.csv")
# # # loads Y as a list
# # # X, Y = ld.load_process_png(label_df)
# # # loads as df
# X, Y = ld.load_process_png_v2(label_df)
#
# Y.to_csv("C:/Users/s161590/Desktop/Data/X_Ray/processed_YYY.csv")
#
# Yclass, Yloc = ld.bind_location_labels("C:/Users/s161590/Desktop/Data/X_Ray/Bbox_List_2017.csv", Y, PATCH_SIZE)
#
# X_tr, X_t, Y_tr, Y_t = ld.split_test_train(X, Y)
# print("*********************SPLIT")
# print((X_tr.shape))
# print((Y_tr.shape))
# print((X_t.shape))
# print((Y_t.shape))
#
#
# X_train, Y_train_df= model.get_feature_extraction(X_tr,Y_tr, batch_size=2, seed=0)
# X_test, Y_test_df  = model.get_feature_extraction(X_t,Y_t, batch_size=2, seed=0)
#
# print("*********************FE")
# print((X_train.shape))
# print((Y_train_df.shape))
# print((X_test.shape))
# print((Y_test_df.shape))
#
# Y_train, Y_test = Y_train_df.to_numpy(), Y_test_df.to_numpy()
# print("*********************x train shape")
# print((X_train.shape))
# print((X_test.shape))
# print((Y_train.shape))
# print((Y_test.shape))
#
# model.model(X_train, Y_train, X_test, Y_test, P=16, start_learning_rate = 0.001,
#           num_epochs=20, num_iter=5,  minibatch_size=5, print_cost=True)


# #############################################
# ######## CODE FOR bounding boxes ############
# #############################################
SERVER_PATH_C ="/home/rnromanova/scripts/Data_Entry_2017.csv"
SERVER_PATH_L ="/home/rnromanova/scripts/Bbox_List_2017.csv"
SERVER_PATH_I = "/home/rnromanova/XRay14/images/batch1"
SERVER_OUT = "/home/rnromanova/scripts/out"

LOCAL_PATH_C = "C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv"
LOCAL_PATH_L = "C:/Users/s161590/Desktop/Data/X_Ray/Bbox_List_2017.csv"
LOCAL_PATH_I = "C:/Users/s161590/Desktop/Data/X_Ray/images"
LOCAL_OUT = "C:/Users/s161590/Desktop/Data/X_Ray/out"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# label_df = ld.get_classification_labels(SERVER_PATH_C, False)
# X, Y = ld.load_process_png_v2(label_df, SERVER_PATH_I)
# Y = ld.couple_location_labels(SERVER_PATH_L , Y, ld.PATCH_SIZE, SERVER_OUT)
#

label_df = ld.get_classification_labels(LOCAL_PATH_C, False)
X, Y = ld.load_process_png_v2(label_df, LOCAL_PATH_I)
Y = ld.couple_location_labels(LOCAL_PATH_L , Y, ld.PATCH_SIZE, LOCAL_OUT)


# separate classification and localization
# then you can specify different percentages for each group
Y_class, Y_loc = ld.separate_localization_classification_labels(Y)

Y_new = ld.keep_only_diagnose_columns(Y)
print(Y_new)

X_train, Y_train_df= model.get_feature_extraction(X,Y_new, batch_size=2, seed=0)
Y_train = ld.reshape_Y(Y_new)


model.build_model(X_train, Y_train, X_train, Y_train, P=16, start_learning_rate = 0.001,
                  num_epochs=20, num_iter=5, minibatch_size=5, print_cost=True)
