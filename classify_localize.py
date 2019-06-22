# import tensorflow as tf
import load_data as ld
import numpy as np



#############################################
######## CODE FOR classification ############
#############################################
# import model
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
label_df = ld.get_classification_labels("C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv", False)
X, Y = ld.load_process_png_v2(label_df)
Y = ld.couple_location_labels("C:/Users/s161590/Desktop/Data/X_Ray/Bbox_List_2017.csv", Y, ld.PATCH_SIZE)
# separate classification and localization
Y_class, Y_loc = ld.separate_localization_classification_labels(Y)

Y_new = ld.keep_only_diagnose_columns(Y)
Y_input = ld.reshape_Y(Y_new)


# print(type(Y_new[:,0]))
ynp = Y_new.to_numpy()
print(len(Y_new.values.tolist()[0][0]))
# print(Y_new[0,:].shape)
print(type(ynp))
print(ynp[0].flatten().shape)
# test2 = ynp[0].shape
test= np.resize(ynp,(21, 16, 16, 14))
print(test[20][0][0][0].shape)
# SPLIT INTO TRAIN AND  TEST  BOTH CLASS AND LOC
# X_tr, X_t, Y_tr, Y_t = ld.split_test_train(X, Y)
newarr = []
