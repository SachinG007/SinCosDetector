import tensorflow as tf 

num_time = 5
batch_size = 1024
lstm_sizes = [128]
keep_prob_ = 0.75 # fixed

num_iterations = 1000
LR = 0.0001		
init_std = 0.1


dense_units = num_labels = 2

fc_size = [64 , 3]

DTYPE = tf.float32

log_dir = './train'
save_path = './saved_model"'

max_gradient_norm = 10
# test_labels_length = 2814

# test_bool = False
# train_bool = True	

# warmup_steps = 0
#How to warmup learning rates. Options include: 
#t2t: Tensor2Tensor's way, start with lr 100 times smaller, then exponentiate until the specified lr.\
# warmup_scheme = "t2t" 
