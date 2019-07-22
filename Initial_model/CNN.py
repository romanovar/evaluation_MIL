import tensorflow as tf
import numpy as np


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    x = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    y = tf.placeholder(tf.float32, shape=(None, n_y))

    return x, y

def initialize_weights_LeNet5():
    tf.set_random_seed(0)
    #todo: think of the filter sizes
    ### [filter_height, filter_width, in_channels, out_channels]: [5,5,1,20]
    W1 = tf.get_variable("W1", [5, 5, 1, 5], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [5, 5, 5, 10], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    ### filter size is equal to input size such the FC is turned into a Conv
    # W3 = tf.get_variable("W3", (156250, 75000), initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1,
                  "W2": W2}
                  # "W3": W3}

    return parameters


def forward_propagation_LeNet5(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']

    # input: (?, 512, 512,1  output:(?, 508 , 508, KernelSize)
    C1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'VALID')
    A1 = tf.nn.relu(C1)
    # input: (?, 508 , 508, KernelSize) output:(?, 254, 254, 50)
    P1 = tf.nn.max_pool(A1, ksize = [1,2, 2,1], strides = [1,2, 2,1], padding = 'VALID')


    # input: (?, 254, 254, 30) output:(?, 250, 250, 50)
    C2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'VALID')
    A2 = tf.nn.relu(C2)
    # input: (?, 250, 250, 30) output: (?, 125, 125, 50)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    ### input: (?, 125, 125, 50) output: 125X125X....=15625X50 =781250
    FC1 = tf.contrib.layers.flatten(P2)
    # in 150,000 out:75,000
    ### TODO: change output kernel size - for testing purposes put on small number
    FC1 = tf.contrib.layers.fully_connected(FC1, 100, activation_fn=tf.nn.relu)
    FC2 = tf.contrib.layers.fully_connected(FC1, 50, activation_fn=tf.nn.relu)

    #output layer
    y_prob = tf.contrib.layers.fully_connected(FC2, 1, activation_fn= None) #tf.nn.sigmoid)
    # y_hat =  tf.greater(out, tf.constant(0.5))

    return C1, P1, C2, P2, FC1, FC2, y_prob


def forward_propagation_Attention(X, parameters):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']


    # input: (?, 512, 512,1  output:(?, 508 , 508, KernelSize)
    C1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'VALID')
    A1 = tf.nn.relu(C1)
    # input: (?, 508 , 508, KernelSize) output:(?, 254, 254, 50)
    P1 = tf.nn.max_pool(A1, ksize = [1,2, 2,1], strides = [1,2, 2,1], padding = 'VALID')


    # input: (?, 254, 254, 30) output:(?, 250, 250, 50)
    C2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'VALID')
    A2 = tf.nn.relu(C2)
    # input: (?, 250, 250, 30) output: (?, 125, 125, 50)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    ### input: (?, 125, 125, 50) output: 125X125X....=15625X50 =781250
    F = tf.contrib.layers.flatten(P2)
    # in 150,000 out:75,000
    FC1 = tf.contrib.layers.fully_connected(F, 50000, activation_fn=tf.nn.relu)


    ### attention model
    att1 = tf.contrib.layers.fully_connected(FC1, 5000, activation_fn=tf.nn.tanh)
    att2 = tf.contrib.layers.fully_connected(att1, 1, activation_fn=None)

    att2 = tf.contrib.layers.fully_connected(att2, 2, activation_fn=tf.nn.sigmoid)
    Pr1 = tf.tensordot(FC1, att2)

    out = tf.contrib.layers.fully_connected(Pr1, 1, activation_fn=tf.nn.sigmoid)

    # y_hat =  tf.greater(out, tf.constant(0.5))
    return C1, P1, C2, P2, FC1, att1, att2, Pr1, out, y_hat


### MAXIMILIAN ILSE CODE
# #     # AUXILIARY METHODS
# def calculate_classification_error(self, X, Y):
#     Y = Y.float()
#     _, Y_hat, _ = self.forward(X)
#     # error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
#     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
#
#     return error, Y_hat
#
#
# # for the loss
# def calculate_objective(self, X, Y):
#     Y = Y.float()
#     Y_prob, _, A = self.forward(X)
#     Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
#     neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
#
#     return neg_log_likelihood, A

def compute_loss_cost(Y_hat, Y):
    # cost = tf.reduce_mean(1-tf.equal(Y_hat, Y))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=Y_hat,name=None))
    return cost

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(512, 512, 1, 1)
    parameters = initialize_weights_LeNet5()
    a1, a2, a3, a4, a5, a6, out = forward_propagation_LeNet5(X, parameters)
    cost = compute_loss_cost(out, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    # Y=
    a = sess.run(cost, {X: np.random.randn(2,512, 512, 1), Y: [[1], [0]]})

    print("cost = " + str(a))
    print("y = " + str(Y))
    print("Z2 = " + str(a2))
    print("Z3 = " + str(a3))
    print("Z4 = " + str(a4))
    print("out = " + str(out))
    print("out = " + str(a))
    print("y = " + str(cost))


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0, n_W0, m, n_y)

    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()

    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X_train, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer.minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, temp_cost = sess.run(cost, optimizer)  # feed_dict={z:logits, y:labels}
                ### END CODE HERE ###

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters