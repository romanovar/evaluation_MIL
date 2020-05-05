from keras.preprocessing.image import load_img, img_to_array, save_img
from cnn.preprocessor.load_data_mura import padding_needed, pad_image
from cnn.keras_utils import image_larger_input, calculate_scale_ratio
import os
import pandas as pd


def create_new_directory(parent_folder, new_folder_name):
    new_path = os.path.join(parent_folder, new_folder_name)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        print("New directory " + new_path + " created")
    else:
        print(new_path + " already exists!")
    return new_path


def decrease_image_size(new_height, new_width, image_dir):
    return load_img(image_dir, target_size=(new_height, new_width), color_mode='rgb')


def resize_image(image_dir, image_new_height, image_new_width, resize_method):
    original_img_width, original_img_height = load_img(image_dir, target_size=None, color_mode='rgb').size
    decrease_needed = image_larger_input(original_img_width, original_img_height, image_new_height, image_new_width)

    # this just decreases the image size to the new image size WITHOUT checking if ratio is kept
    # this is used only for xray dataset where images are 1024x1024
    if resize_method:
        resized_image = decrease_image_size(image_new_height, image_new_width, image_dir)

    else:
        # IF one or both sides of the image have bigger size than the requires input, then decrease is needed
        # rescaling with preserving the image ratio
        if decrease_needed:
            ratio = calculate_scale_ratio(original_img_width, original_img_height, image_new_width,
                                          image_new_height)
            assert ratio >= 1.00, "wrong ratio - it will increase image size"
            assert int(original_img_height / ratio) == image_new_height or \
                   int(original_img_width / ratio) == image_new_width, "error in computation"

            resized_image = load_img(image_dir,
                                     target_size=(int(original_img_height / ratio),
                                                  int(original_img_width / ratio)),
                                     color_mode='rgb')
        else:
            # ELSE just open image in its original form
            resized_image = load_img(image_dir, target_size=None, color_mode='rgb')

        ### PADDING
        pad_needed = padding_needed(resized_image)

        if pad_needed:
            resized_image = pad_image(resized_image, final_size_x=image_new_width, final_size_y=image_new_height)

    return resized_image


def preprocess_images_from_dataframe(df, image_new_height, image_new_width, resize_method, parent_folder,
                                     new_folder_name, df2):

    processed_images_dir = create_new_directory(parent_folder, new_folder_name)
    new_df = df.copy()

    for index, row in df.iterrows():
        image_dir = row['Dir Path']
        image_name = os.path.split(image_dir)[-1]
        resized_image = resize_image(image_dir, image_new_height, image_new_width, resize_method)
        img_array = img_to_array(resized_image)

        save_img(processed_images_dir+'/'+image_name, img_array)
        df2.loc[df2['Image Index'] == image_name, 'Dir Path'] =  processed_images_dir+'/'+image_name
        new_df.loc[index]['Dir Path'] = processed_images_dir+'/'+image_name
    new_df.to_csv(processed_images_dir+'/processed_'+new_folder_name+'.csv')
    return new_df, df2


def fetch_preprocessed_images_csv(parent_folder, new_folder_name):
    new_path = os.path.join(parent_folder, new_folder_name)
    assert os.path.exists(new_path), " Directory not found. Please, run preprocess_images.py first"
    if os.path.exists(new_path):
        return pd.read_csv(new_path+'/processed_'+new_folder_name+'.csv', index_col=0)


def combine_preprocessed_csv(df_train, df_test, df_val):
    return pd.concat([df_train, df_val, df_test])