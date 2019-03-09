import tensorflow as tf  
import numpy as np 
import os 
import sys
# sys.path.insert(0, '../data')
# import utils
# import pandas as pd
import random
import hparams
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, hparams.DTYPE, tf.truncated_normal_initializer(stddev=hparams.init_std)) # function to intilaize weights for each layer

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, hparams.DTYPE, tf.constant_initializer(0.1, dtype=tf.float32)) # fucntion to intiliaze bias vector for each layer

def getbatch_X(step, X):
    # X = X.values
    data = np.zeros((hparams.batch_size, hparams.num_time, 1))
    for i in range(hparams.batch_size):
        for j in range(hparams.num_time):
            data[i][j] = X[step*hparams.batch_size + i][j]
    return data   

def getbatch_labels(step, labels):
    data = np.zeros(hparams.batch_size)
    for i in range(hparams.batch_size):
        data[i] = labels[step*hparams.batch_size + i]
    return data   

def build_lstm_layers(lstm_sizes, inputs, keep_prob_, batch_size):

    # lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(128)
    # # Add dropout to the cell
    # # drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    # # Stack up multiple LSTM layers, for deep learning
    # # cell = tf.contrib.rnn.MultiRNNCell(drops)
    # # Getting an initial state of all zeros
    # initial_state = lstm.zero_state(batch_size, tf.float32)
    # lstm_outputs, final_state = tf.nn.dynamic_rnn(lstm, inputs, initial_state=initial_state)
    # return lstm_outputs, final_state

    cell = tf.contrib.rnn.LSTMCell(128, forget_bias=1.0)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob_)
    lstm_output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    return lstm_output, final_state

def apply_dense_layer(inputs):
    logits = tf.layers.dense(inputs, hparams.dense_units)
    return logits


def fully_connected(size, prev_layer, name_scope):
    with tf.variable_scope(name_scope) as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, size])
        biases = _bias_variable('biases', [size])
        output = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
    return output

def core_model(lstm_sizes, inputs, keep_prob_, batch_size):
    outputs, state = build_lstm_layers(lstm_sizes, inputs, keep_prob_, batch_size)
    # import pdb;pdb.set_trace()
    outputs = outputs[:,-1,:]
    prev_layer = outputs
    fc_size = hparams.fc_size
    
    next_layer = []
    for l in range(len(fc_size)):    
        next_layer = fully_connected(fc_size[l], prev_layer, 'FC_' + str(l))
        prev_layer = next_layer

    logits = next_layer
    logits = apply_dense_layer(outputs)
    
    return logits,outputs


def compute_loss(gt_labels, logits):    
    
    # import pdb;pdb.set_trace()
    print(gt_labels)
    print(logits)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gt_labels, logits=logits)

    return crossent	

def train():

    
    input_data = tf.placeholder(tf.float32, [hparams.batch_size, hparams.num_time, 1])
    input_class  = tf.placeholder(tf.int32, [hparams.batch_size])
    

    prediction,only_l_out = core_model(hparams.lstm_sizes, input_data, hparams.keep_prob_, hparams.batch_size)

    loss = compute_loss(input_class, prediction)
    loss = tf.reduce_mean(loss)
    # test time accuracy calculation
    # prediction = tf.argmax(logits, 1)
    # # prediction_test = tf.argmax(logits_test,2)

    # equality = tf.equal(prediction, correct_answer)
    # accuracy = np.sum(tf.cast(equality, tf.float32))/hparams.batch_size
    
    # tf.summary.scalar('accuracy', tf.reduce_mean(accuracy))
    

    with tf.name_scope("compute_gradients"):
        # compute_gradients` returns a list of (gradient, variable) pairs
        params = tf.trainable_variables()
        # import pdb;pdb.set_trace()
        i = 0
        # for var in params:
            # print("kfbffeljalfbaljefalefnlaeknaklnflaeneglknealgkenalaenlaekgnal",i)
            # i = i + 1
            # tf.summary.histogram(var.name, var)
        
        grads = tf.gradients(xs=params, ys=loss, colocate_gradients_with_ops=True)    # optimizer.compute_gradients(loss)
        # clipped_grads = grads
        clipped_grads, grad_norm_summary, grad_norm = gradient_clip(grads, max_gradient_norm=hparams.max_gradient_norm)
        grad_and_vars = zip(clipped_grads, params)

    
    # lr = learning_rate = tf.constant(hparams.LR)
    # # warm-up
    # learning_rate = _get_learning_rate_warmup(global_step, lr)
    # # decay
    # learning_rate = _get_learning_rate_decay(hparams)
    
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(hparams.LR)
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
        train_writer = tf.summary.FileWriter(hparams.log_dir ,sess.graph)
        global test_flag
        # saver = tf.train.Saver()            

        with open("input_data.txt", "rb") as fp:
            X = pickle.load(fp)

        with open("input_data_labels.txt","rb") as fp:
            true_labels = pickle.load(fp)
            # import pdb;pdb.set_trace()

        for j in range(hparams.num_iterations): 
            
            print ("Training:: iteration: ", j)
            test_flag = 0

            length = len(true_labels)
            train_length = int(length*0.8)                      # Train dataset length 
            test_length = int(length*0.2)                       # Test dataset length 
            
            X_train = X[0:train_length]                         # Train dataset (features)
            X_test = X[-test_length:]                           # Test dataset (features)
                
            true_labels_train = true_labels[0:train_length]                   # Train dataset (labels)
            true_labels_test = true_labels[-test_length:]                     # Test dataset (labels)    
            

            epochs = train_length/hparams.batch_size         # number of epochs for training 
            
            avg_cost = 0.0

            for i in range(epochs):
                print ("Training:: Epoch ", i)
                
                train_batch_input = getbatch_X(i, X_train)	 
                train_batch_labels = getbatch_labels(i, true_labels_train)    
                train_batch_labels = np.reshape(train_batch_labels,(hparams.batch_size))

                _, loss_val, pred, l_pred, summary = sess.run(
                    [apply_gradient_op, loss, prediction, only_l_out, merged],
                    feed_dict={
                        input_data: train_batch_input, 
                        input_class: train_batch_labels,
                    }
                ) 
                print(pred)
                print(l_pred)
                print(np.shape(train_batch_input))
                train_writer.add_summary(summary)
                # print pred*np.sqrt(1315.2341209347906) + 198.05518950451153
                avg_cost += loss_val 
                print ("loss: ", loss_val)

            print ("Average Cost: ", avg_cost/epochs )
            save_path = saver.save(sess, './saved_model')
            print ("Model saved")    
            # total_iter_test_error = 0

train()




