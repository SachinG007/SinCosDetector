import numpy as np
import random
import tensorflow as tf
import pickle
import tensorflow.contrib.layers as layers
import os 
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="1"

NUM_BITS = 1000
INPUT_SIZE    = 1       
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 3      
TINY          = 1e-6   
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TEST_EPOCHS = 10
max_gradient_norm = 10

def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm

def compute_loss(gt_labels, logits):    
    
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gt_labels, logits=logits)

    return crossent 

def generate_batch(step, num_bits, batch_size):

    x = np.empty((num_bits, batch_size, 1))
    y = np.empty((batch_size))

    with open("input_data.txt", "rb") as fp:
        X = pickle.load(fp)

    with open("input_data_labels.txt","rb") as fp:
        true_labels = pickle.load(fp)

    for i in range(batch_size):
        # for j in range(num_bits):        
        x[:, i, 0] = X[step * batch_size + i]
        y[i] = true_labels[step * batch_size + i]
    return x, y

def generate_test_batch(num_bits, batch_size):

    x = np.empty((num_bits, batch_size, 1))
    y = np.empty((batch_size))

    with open("input_data.txt", "rb") as fp:
        X = pickle.load(fp)

    with open("input_data_labels.txt","rb") as fp:
        true_labels = pickle.load(fp)

    for i in range(batch_size):
        # for j in range(num_bits):        
        x[:, i, 0] = X[9000 - 1 + i]
        y[i] = true_labels[9000 - 1 + i]
    return x, y

def train():

    inputs  = tf.placeholder(tf.float32, (NUM_BITS, BATCH_SIZE, INPUT_SIZE))  # (time, batch, in)
    correct_outputs = tf.placeholder(tf.int64, (BATCH_SIZE)) # ( batch)

    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

    initial_state = cell.zero_state(BATCH_SIZE, tf.float32)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
    rnn_outputs_last = np.zeros((BATCH_SIZE,RNN_HIDDEN))
    rnn_outputs_last = rnn_outputs[-1:]
    rnn_outputs_last_r = tf.reshape(rnn_outputs_last,[BATCH_SIZE,RNN_HIDDEN])
    # import pdb;pdb.set_trace()

    predicted_outputs = tf.contrib.layers.fully_connected(rnn_outputs_last_r, OUTPUT_SIZE)

    #test error part
    correct_prediction = tf.equal(tf.argmax(predicted_outputs, 1), correct_outputs)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = compute_loss(correct_outputs,predicted_outputs)
    loss = tf.reduce_mean(loss)

    with tf.name_scope("compute_gradients"):
        params = tf.trainable_variables()
        grads = tf.gradients(xs=params, ys=loss, colocate_gradients_with_ops=True)    # optimizer.compute_gradients(loss)
        clipped_grads, grad_norm_summary, grad_norm = gradient_clip(grads, max_gradient_norm=max_gradient_norm)
        grad_and_vars = zip(clipped_grads, params)


    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    apply_gradient_op = optimizer.apply_gradients(grad_and_vars, global_step)

    tf.summary.scalar('loss', loss)

    saver = tf.train.Saver()
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        )
    session_config.gpu_options.allow_growth = True
    
    sess = tf.InteractiveSession(config=session_config)

    initializer = tf.contrib.layers.xavier_initializer()


    with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
        # with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
        init = tf.global_variables_initializer()
        sess.run(init)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train' ,sess.graph)

        ITERATIONS_PER_EPOCH = 100#int(10000/BATCH_SIZE) - 1
        NUM_EPOCHS = 100

    for epoch in range(NUM_EPOCHS):
        print("Epoch Number: ",epoch)
        
        epoch_error_t = 0
        
        for step in range(ITERATIONS_PER_EPOCH):

            x, y = generate_batch(step, num_bits=NUM_BITS, batch_size=BATCH_SIZE)
            
            _, loss_val, summary = sess.run(
                [apply_gradient_op, loss, merged],
                feed_dict={
                    inputs: x, 
                    correct_outputs: y,
                }
            ) 

            # print("step: ", step, "loss: ",loss_val)
            epoch_error_t = epoch_error_t + loss_val
            train_writer.add_summary(summary)
        epoch_error_t /= ITERATIONS_PER_EPOCH
        print("Epoch Number: ", epoch, "Train Error :", epoch_error_t)
        save_path = saver.save(sess, './saved_model')


        #Test Accuracy Calc
        for k in range(TEST_EPOCHS):
            test_accuracy = 0
            x, y = generate_test_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)    
            test_accuracy += sess.run(
                [accuracy],
                feed_dict={
                    inputs: x, 
                    correct_outputs: y,
                }
            )[0]
        test_accuracy = test_accuracy / TEST_EPOCHS
        print("Epoch Number: ", epoch, "Test Acc :", test_accuracy)

train()
    