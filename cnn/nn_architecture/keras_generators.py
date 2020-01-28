"""
inspired by https://github.com/neuralmed/learning_with_bbox
"""
import cv2
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
from cnn.preprocessor.load_data_mura import padding_needed, pad_image
from cnn.keras_utils import process_loaded_labels, image_larger_input, calculate_scale_ratio


class BatchGenerator(Sequence):
    def __init__(self, instances, batch_size=16, shuffle=True,
                 norm=None, net_h=512, net_w=512, box_size=16, processed_y = None, interpolation=True):

        self.instances = instances
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.norm = norm
        self.net_h = net_h
        self.net_w = net_w
        self.box_size = box_size
        self.processed_y = processed_y
        self.interpolation = interpolation

        if shuffle: np.random.shuffle(self.instances)

    def __len__(self):
        # return int(np.ceil(float(len(self.instances)) / self.batch_size))
        return int(np.floor(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, self.net_w, self.net_h, 3))  # input images
        y_batch = np.zeros((r_bound - l_bound, self.box_size, self.box_size, 14))
        y_batch = np.zeros((r_bound - l_bound, self.box_size, self.box_size, 1))

        instance_count = 0
        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            image_dir = train_instance[0]
            img_width, img_height = load_img(image_dir, target_size=None, color_mode='rgb').size
            decrease_needed = image_larger_input(img_width, img_height, self.net_w, self.net_h)


            if self.interpolation:
                #### NEAREST INTERPOLATION
                image = img_to_array(
                    load_img(image_dir, target_size=(self.net_h, self.net_w), color_mode='rgb'))
            else:
                # IF one or both sides have bigger size than the input, then decrease is needed
                if decrease_needed:
                    ratio = calculate_scale_ratio(img_width, img_height, self.net_w, self.net_h)
                    assert ratio >= 1.00, "wrong ratio - it will increase image size"
                    assert int(img_height/ratio) == self.net_h or int(img_width/ratio) == self.net_w, \
                        "error in computation"
                    image = img_to_array(load_img(image_dir, target_size=(int(img_height/ratio), int(img_width/ratio)),
                                                  color_mode='rgb'))
                else:
                    #ELSE just open image in its original form
                    image = img_to_array(load_img(image_dir, target_size=None, color_mode='rgb'))
                ### PADDING
                pad_needed = padding_needed(image)

                if pad_needed:
                    image = pad_image(image, final_size_x=self.net_w, final_size_y=self.net_h)

            if self.norm != None:
                x_batch[instance_count] = self.norm(image)
            else:
                x_batch[instance_count] = image

            train_instances_classes = []

            if self.processed_y is not None:
                for i in range(1, train_instance.shape[0]):  # (15)
                    if self.processed_y:
                        g = process_loaded_labels(train_instance[i])
                        train_instances_classes.append(g)
                    else:
                        # labels = np.fromstring(train_instance[i], dtype=int, sep=' ')
                        # train_instances_classes.append(train_instance[i])
                        labels = process_loaded_labels(train_instance[i])

                        train_instances_classes.append(labels)
                y_batch[instance_count] = np.transpose(np.asarray(train_instances_classes), [1, 2, 0])
            else:
                y_batch[instance_count]= None

            instance_count += 1
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)

    def load_image(self, i):
        image_name = self.instances[i]

        image = img_to_array(load_img(image_name, target_size=(self.net_w, self.net_h), color_mode='rgb'))
        return image

    def get_batch_image_indices(self, idx):
        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        return self.instances[l_bound:r_bound][:, 0]
