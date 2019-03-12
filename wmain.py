import numpy as np
import random
import tensorflow as tf
import pickle
import tensorflow.contrib.layers as layers
import os 
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=int, default=0, help="do you want to test only")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--test_iters", type=int, default=7, help="Test Epocs")
parser.add_argument("--learning_rate", type = float , default=0.0001, help="LEARNING_RATE")
parser.add_argument("--num_epochs", type = int , default=100, help="Num Epochs")
a = parser.parse_args()

NUM_BITS = 50
INPUT_SIZE    = 1       
RNN_HIDDEN    = 32
FC1_OUT_SIZE = 16
OUTPUT_SIZE   = 3      
TINY          = 1e-6   
LEARNING_RATE = a.learning_rate
BATCH_SIZE = a.batch_size
TEST_ITERS = a.test_iters
ITERATIONS_PER_EPOCH = 65#int(20000/BATCH_SIZE) - 1
NUM_EPOCHS = a.num_epochs
max_gradient_norm = 10
only_testing = a.test

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

def generate_test_batch(step, num_bits, batch_size):

    x = np.empty((num_bits, batch_size, 1))
    y = np.empty((batch_size))

    # with open("input_data.txt", "rb") as fp:
    with open("test_data.txt", "rb") as fp:    
        X = pickle.load(fp)

    # with open("input_data_labels.txt","rb") as fp:
    with open("test_data_labels.txt", "rb") as fp:    
        true_labels = pickle.load(fp)

    for i in range(batch_size):
        # for j in range(num_bits):        
        # x[:, i, 0] = X[18000 - 1 + step*batch_size + i]
        x[:, i, 0] = X[step*batch_size + i]
        # y[i] = true_labels[18000 - 1 + step*batch_size + i]
        y[i] = true_labels[step*batch_size + i]
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

    fc1_outputs = tf.contrib.layers.fully_connected(rnn_outputs_last_r, FC1_OUT_SIZE)
    # fc1_outputs_relu = tf.nn.relu(fc1_outputs)
    predicted_outputs = tf.contrib.layers.fully_connected(fc1_outputs, OUTPUT_SIZE)

    #test error part
    predicts = tf.argmax(predicted_outputs, 1)
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
        
        if only_testing == 0:
            sess.run(init)
        else:
            saver.restore(sess, "./tmp2/model.ckpt")
            print("Loaded saved model")

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train' ,sess.graph)



    epoch_error_list = []
    test_accuracy_list = []
  

    for epoch in range(NUM_EPOCHS):
        print("Epoch Number: ",epoch)
        
        epoch_error_t = 0
        
        for step in range(ITERATIONS_PER_EPOCH):

            x, y = generate_batch(step, num_bits=NUM_BITS, batch_size=BATCH_SIZE)
            
            _, loss_val, summary, predicts_o = sess.run(
                [apply_gradient_op, loss, merged, predicted_outputs],
                feed_dict={
                    inputs: x, 
                    correct_outputs: y,
                }
            ) 
            # print(y)
            # print("predicted")
            print(predicts_o)
            # print("step: ", step, "loss: ",loss_val)
            epoch_error_t = epoch_error_t + loss_val
            train_writer.add_summary(summary)
        epoch_error_t /= ITERATIONS_PER_EPOCH
        epoch_error_list.append(epoch_error_t)
        print("Epoch Number: ", epoch, "Train Error :", epoch_error_t)
        
        save_path = saver.save(sess, './tmp2/model.ckpt')
        print("Model Saved")


        #Test Accuracy Calc
        test_accuracy = 0
        for k in range(TEST_ITERS):
            
            x, y = generate_test_batch(k, num_bits=NUM_BITS, batch_size=BATCH_SIZE)    
            test_accuracy_temp, predicts_o = sess.run(
                [accuracy, predicts],
                feed_dict={
                    inputs: x, 
                    correct_outputs: y,
                }
            )

            test_accuracy += test_accuracy_temp
            print(y)
            print("Preicted Labels")
            print(predicts_o)


        test_accuracy = test_accuracy / TEST_ITERS
        test_accuracy_list.append(test_accuracy)
        print("Epoch Number: ", epoch, "Test Acc :", test_accuracy)


        with open('Train_error.txt', 'w') as f:   

            for item in epoch_error_list:
                f.write("%s\n"%item)

        with open('Test_Acc.txt', 'w') as f:   
            for item in test_accuracy_list:
                f.write("%s\n"%item)

train()
    