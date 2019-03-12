import matplotlib.pyplot as plt
import numpy as np
import pylab
import pickle
import random
 

with open("test_data.txt", "rb") as fp:
    X = pickle.load(fp)
	
sample = 1000
x = np.arange(sample)
y = X[0]
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# import pdb;pdb.set_trace()
# print(sliding_window)