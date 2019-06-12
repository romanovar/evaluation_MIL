import tensorflow as tf
import numpy as np
from tensorflow.contrib.gan.python.eval import preprocess_image

from tensorflow.python.tools import inspect_checkpoint as chkp
import ReadXray as rxr




images_X = rxr.find_load_annotated_png_files(path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images")
print(images_X.shape)
images_X = np.reshape(images_X, (-1, 1024, 1024, 1))
print(images_X.shape)
input_ten = tf.placeholder(tf.float32, shape=(None, 1024, 1024, 1))

#
# export_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/'
# pb_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/saved_model.pb'
#
# ckpt_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_50_2017_04_14.tar/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'

# # saver=tf.train.Saver()
# with tf.Session() as sess:
#     chkp.print_tensors_in_checkpoint_file(ckpt_dir, tensor_name='', all_tensors = True, all_tensor_names=True)

    # saver.restore(sess, ckpt_dir)
#
# with tf.Session() as sess:
#     y_prepr = preprocess_image(input_ten,height =299,width=299)
#     result_prepro = sess.run(y_prepr, {input_ten: images_X})
#     print(result_prepro.shape)
#
#     meta_graph_def = tf.saved_model.loader.load(sess, ['train'], export_dir)
#     graph = tf.get_default_graph()
#     op = graph.get_operations()
#     for m in op:
#         print(m.values())
#     # print("***************************")
#     # print(graph.get_tensor_by_name("input_tensor:0"))
#     # print(graph.get_tensor_by_name("resnet_model/Relu_48:0"))
#     X = graph.get_tensor_by_name("input_tensor:0")
#     Y = graph.get_tensor_by_name("resnet_model/Relu_48:0")
#     results= sess.run(Y, {X: result_prepro})
#     print(results.shape)
# #
# #     results  = sess.run(Y, {input_ten:input_X})
# #     print(results.shape)

export_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/'
pb_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/saved_model.pb'

########################################## load from CHECKPOINT #############################################

export_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/'
pb_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/saved_model.pb'

ckpt_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_50_2017_04_14.tar/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'


######################################## LOAD SAVEDMODEL ####################################################


export_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/'
pb_dir = 'C:/Users/s161590/Desktop/Data/resnet_v2_fp32_savedmodel_NHWC.tar/resnet_v2_fp32_savedmodel_NHWC/resnet_v2_fp32_savedmodel_NHWC/1538687283/saved_model.pb'

images_X = rxr.find_load_annotated_png_files(path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images")

input = tf.placeholder(tf.float32, shape=(None, 1024, 1024, 1))
input_X = np.random.randn(64, 224, 224, 3)

graph = tf.Graph()
with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    signature = meta_graph_def.signature_def
    print(signature)
    graph = tf.get_default_graph()
    op = graph.get_operations()
    for m in op:
        print(m.values())

    # batch_size_placeholder = graph.get_tensor_by_name('batch_size_placeholder:0')
    # print(batch_size_placeholder)
    #
    # features_placeholder = graph.get_tensor_by_name('features_placeholder:0')
    # print(features_placeholder)
    #
    # labels_placeholder = graph.get_tensor_by_name('labels_placeholder:0')
    # print(labels_placeholder)
    #
    # prediction = graph.get_tensor_by_name('dense/BiasAdd:0')
    # print(prediction)


    # sess.run(prediction, feed_dict={
    #     batch_size_placeholder: some_value,
    #     features_placeholder: some_other_value,
    #     labels_placeholder: another_value
    # })
    X = graph.get_tensor_by_name("input_tensor:0")
    Y = graph.get_tensor_by_name("resnet_model/Relu_48:0")
    results  = sess.run(Y, {X:input_X})
    results = sess.run(X)

    print(results)

#####################################IMAGE PREPROCESSING ######################################
# def format_example(images):
#     for image in images:
#       image = tf.cast(image, tf.float32)
#       image = (image/127.5) - 1
#       image = tf.image.resize(image, (224, 224))
#     return image
#

################################################### RESNET - NOT TRAINED ON IMAGENET ########################

# import tensorflow as tf
# from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50
#
# # import tensorflow.contrib.slim.python.slim.nets
#
# input = tf.placeholder(tf.float32, shape=(None, 1024, 1024, 1))
# # blocks = [ resnet.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2), resnet.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
# #            resnet.resnet_v2_block('block3', base_depth=256, num_units=23, stride=2), resnet.resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)]
#
#
# resnet_v2_50(input,  is_training=True, global_pool=False)
#
