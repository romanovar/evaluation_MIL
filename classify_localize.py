import load_data as ld
import model
import os


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

Y_patch_labels = ld.keep_only_diagnose_columns(Y)
print(Y_patch_labels)

Y_img_label = ld.keep_only_classification_columns(Y)
# model.fine_tune_resnet(X, Y_img_label)

X_train, Y_train_df= model.get_feature_extraction(X,Y_patch_labels, batch_size=2, seed=0)
Y_train = ld.reorder_Y(Y_patch_labels)


model.build_model(X_train, Y_train, X_train, Y_train, P=16, start_learning_rate = 0.001,
                  num_epochs=50, num_iter=5, minibatch_size=5, print_cost=True)
