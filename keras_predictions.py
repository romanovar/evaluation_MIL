from keras.models import load_model
import custom_loss as cl
import load_data as ld
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

ON_SERVER = False
IMAGE_SIZE = 512
BATCH_SIZE = 8
BOX_SIZE = 16


SERVER_PATH_C ="/home/rnromanova/scripts/Data_Entry_2017.csv"
SERVER_PATH_L ="/home/rnromanova/scripts/Bbox_List_2017.csv"
SERVER_PATH_I = "/home/rnromanova/XRay14/images/batch1"
SERVER_OUT = "/home/rnromanova/scripts/out"

LOCAL_PATH_C = "C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv"
LOCAL_PATH_L = "C:/Users/s161590/Desktop/Data/X_Ray/Bbox_List_2017.csv"
LOCAL_PATH_I = "C:/Users/s161590/Desktop/Data/X_Ray/images/"
LOCAL_OUT = "C:/Users/s161590/Desktop/Data/X_Ray/out"

IMG_DIR = None
if ON_SERVER:
    label_df = ld.get_classification_labels(SERVER_PATH_C, False)
    X, Y = ld.load_process_png_v2(label_df, SERVER_PATH_I)
    Y = ld.couple_location_labels(SERVER_PATH_L , Y, ld.PATCH_SIZE, SERVER_OUT)
    IMG_DIR = SERVER_PATH_I
else:
    label_df = ld.get_classification_labels(LOCAL_PATH_C, False)
    _, Y = ld.load_process_png_v2(label_df, LOCAL_PATH_I)
    Y = ld.couple_location_labels(LOCAL_PATH_L , Y, ld.PATCH_SIZE, LOCAL_OUT)
    IMG_DIR = LOCAL_PATH_I

XY = ld.keep_index_and_diagnose_columns(Y)

def normalize(im):
    im = im / 255
    return im


model = load_model('model_nodule_bbox_10epochs.h5', custom_objects={'loss_bb': cl.keras_loss}, compile=False)


tes = XY[XY['Image Index']=='00000211_019.png']
patches_ground_truth = np.asarray(tes['Cardiomegaly_loc'])[0]
print(tes['Image Index'][865])
print(np.asarray(tes['Cardiomegaly_loc'])[0])
# image = (load_img(IMG_DIR+''+tes['Image Index'][865], target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='rgb'))
# print(type(image))
image = np.array([img_to_array(load_img(IMG_DIR+''+tes['Image Index'][865],
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='rgb'))])
image = normalize(image)
patch = model.predict(image)
print(patch.shape)
im = plt.imread(IMG_DIR+''+XY['Image Index'][865])
plt.imshow(im, 'bone')
plt.figure()
plt.imshow(patch[0,:,:,1])
plt.figure()
plt.imshow(patches_ground_truth)
plt.show()
