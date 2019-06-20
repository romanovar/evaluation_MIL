import model
import load_data as ld
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.applications import ResNet50V2
from keras.layers import Input
import tensorflow as tf
from tensorflow.python.ops.nn_ops import max_pool
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from keras import backend as K
import pandas as pd




def get_feature_extraction(images, labels, batch_size, seed=0):
    K.clear_session()
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)

    seed = seed + 1
    concat_labels =pd.DataFrame()
    X = np.empty((0, 16, 16, 2048))

    model = ResNet50V2(include_top=False, weights='imagenet',  input_shape=(512, 512, 3))
    model._make_predict_function()
    minibatches = mini_batches(images, labels, batch_size, seed)

    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        fe_mat = model.predict(minibatch_X)
        # feat_extr_X.append((np.asarray(fe_mat).reshape((-1, 16, 16, 2048))))
        X = np.concatenate((X,np.asarray(fe_mat)), axis=0 )

        # IF minibatch does not shuffle, then this line is redundant
        concat_labels = pd.concat([concat_labels, minibatch_Y])

    return X, concat_labels


def slice_image_to_patches(output_fe, P=16):
    if P == 16:
        MP = max_pool(output_fe, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        return MP
    elif P == 12:
        MP = max_pool(output_fe, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
        return MP


# ## PATCH SLICE of 16x16
# P1 = max_pool(x_fe, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
# # PATCH SLICE of 12x12
# P2 = max_pool(x_fe, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
#
def initialize_weights():
    tf.set_random_seed(0)

    W1 = tf.get_variable("W1", [3, 3, 2048, 512], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [1, 1, 512, 14], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    weights = {"W1": W1,
               "W2": W2}

    return weights


def recognition_network(P, weights):
    W1 = weights['W1']
    W2 = weights['W2']

    C1 = tf.nn.conv2d(P, W1, strides=[1, 1, 1, 1], padding='SAME')
    BN1 = tf.layers.batch_normalization(C1)
    A1 = tf.nn.relu(BN1)
    C2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding='SAME')
    return C2


def forward_propagation(X, weights, P):
    MP = slice_image_to_patches(X, P)
    pred = recognition_network(MP, weights)
    return pred


def mini_batches(X, Y, mini_batch_size=5, seed=0, random=False):
    np.random.seed(seed)
    m = X.shape[0]  # number of training examples
    print("m: ")
    print(type(m))
    mini_batches = []

    if random:
        permutation = list(np.random.permutation(m))
    else:
        permutation = list(range(0, m))
        print(permutation)
    shuffled_X = np.take(X, permutation, axis=0)
    shuffled_Y = np.take(Y, permutation, axis=0)

    # shuffled_X = X[:, permutation]
    # shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = (np.floor(m / mini_batch_size)).astype(int)

    print("# batches:")
    print(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = np.take(shuffled_X, range( k * mini_batch_size, (k + 1) * mini_batch_size), axis=0)
        mini_batch_Y = np.take(shuffled_Y, range( k * mini_batch_size, (k + 1) * mini_batch_size), axis=0)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = np.take(shuffled_X, range(mini_batch_size * num_complete_minibatches, m), axis=0)
        mini_batch_Y = np.take(shuffled_Y, range(mini_batch_size * num_complete_minibatches, m), axis=0)

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def random_mini_batches(X, Y, mini_batch_size=5, seed=0):
    np.random.seed(seed)
    m = X.shape[0]  # number of training examples

    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = np.take(X, permutation, axis=0)
    shuffled_Y = np.take(Y, permutation, axis=0)

    # shuffled_X = X[:, permutation]
    # shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = tf.cast(tf.floor(m / mini_batch_size), tf.int32)
    for k in range(0, num_complete_minibatches.eval()):
        start_idx = k * mini_batch_size
        print(start_idx)
        end_idx = (k + 1) * mini_batch_size
        # mini_batch_X = shuffled_X[:, start_idx: end_idx]
        # mini_batch_Y = shuffled_Y[:, start_idx: end_idx]
        mini_batch_X = np.take(shuffled_X, range(start_idx, end_idx), axis=0)
        mini_batch_Y = np.take(shuffled_Y, range(start_idx, end_idx), axis=0)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        start_idx = mini_batch_size * num_complete_minibatches.eval()
        print(start_idx)
        print(mini_batch_size)
        print(num_complete_minibatches.eval())
        print(m)
        mini_batch_X = np.take(shuffled_X, range(start_idx, m), axis=0)
        mini_batch_Y = np.take(shuffled_Y, range(start_idx, m), axis=0)
        # mini_batch_X = shuffled_X[:, (mini_batch_size * num_complete_minibatches):m]
        # mini_batch_Y = shuffled_Y[:, (mini_batch_size * num_complete_minibatches):m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def compute_image_label_classification(nn_output, P):
    subtracted_prob = 1 - nn_output
    ### KEEP the dimension for observations and for the classes
    flat_mat = tf.reshape(subtracted_prob, (-1, P* P, 14))
    min_val = tf.reshape(tf.reduce_min(flat_mat, axis=1), (-1, 1, 14))
    max_val = tf.reshape(tf.reduce_max(flat_mat, axis=1), (-1, 1, 14))
    ## Normalization between [a, b]
    ### ( (b-a) (X - MIN(x)))/ (MAX(x) - Min(x)) + a

    normalized_mat = ((1 - 0.98) * (flat_mat - min_val) /
                      (max_val - min_val)) + 0.98

    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return (tf.cast(1, tf.float32) - element_product)


def compute_loss(nn_output, y_true, P):
    epsilon = tf.pow(tf.cast(10, tf.float32), -7)
    y_hat = compute_image_label_classification(nn_output, P)

    ## IF image does not have bounding box
    loss_classification = - (y_hat * (tf.log(y_true+epsilon))) - ((1-y_hat)*(tf.log(1-y_true+epsilon)))

    return loss_classification, y_hat


def loss_L2(Y_hat,Y, P, L2_rate=0.01):
    normal_loss, image_prob = compute_loss(Y_hat, Y, P)
    # normal_loss = compute_image_label_classification_v2(Y_hat, Y, P)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return normal_loss + L2_rate * sum(reg_losses), image_prob


# TODO: 1) batch size of 5 - DONE
# TODO: 2) train the model with 500k iterations of minibatch - DONE
# TODO: 3) decay the learning rate by 0.1 from 0.001 every 10 epochs of training data - DONE
# TODO: 4) add L2 regularization to the loss function to prevent overfitting - DONE
# TODO: 5) optimize the model by Adam - DONE
# todo: 6) with asynchronous training
# TODO: to change number of iterations
def model(X_train, Y_train, X_test, Y_test, P=16, start_learning_rate = 0.001,
          num_epochs=200, num_iter=500000,  minibatch_size=5, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    costs = []

    X = tf.placeholder(tf.float32, shape=(None, P, P, 2048))
    Y = tf.placeholder(tf.float32, shape=(None, 14))
    weights = initialize_weights()

    Z3 = forward_propagation(X, weights, P)

    loss_Test, image_prob = loss_L2(Z3, Y, P)
    cost = tf.reduce_mean((Y, loss_Test))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               10, 0.1, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):

            minibatch_cost = 0.


            num_minibatches = int(X_train.shape[0] / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            # minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            minibatches = mini_batches(X_train, Y_train, minibatch_size, seed, random=True)


            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                for iter in range(num_iter):
                    _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    minibatch_cost += temp_cost / (num_minibatches*num_iter)

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(image_prob, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, weights


label_df = ld.get_classification_labels(label_dir="C:/Users/s161590/Desktop/Data/X_Ray/test.csv")
# loads Y as a list
# X, Y = ld.load_process_png(label_df)
# loads as df
X, Y = ld.load_process_png_v2(label_df)

X_tr, X_t, Y_tr, Y_t = ld.split_test_train(X, Y)
print("*********************SPLIT")
print((X_tr.shape))
print((Y_tr.shape))
print((X_t.shape))
print((Y_t.shape))


X_train, Y_train_df= get_feature_extraction(X_tr,Y_tr, batch_size=2, seed=0)
X_test, Y_test_df  = get_feature_extraction(X_t,Y_t, batch_size=2, seed=0)

print("*********************FE")
print((X_train.shape))
print((Y_train_df.shape))
print((X_test.shape))
print((Y_test_df.shape))

Y_train, Y_test = Y_train_df.to_numpy(), Y_test_df.to_numpy()
print("*********************x train shape")
print((X_train.shape))
print((X_test.shape))
print((Y_train.shape))
print((Y_test.shape))

model(X_train, Y_train, X_test, Y_test, P=16, start_learning_rate = 0.001,
          num_epochs=20, num_iter=5,  minibatch_size=5, print_cost=True)