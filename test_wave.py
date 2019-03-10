import matplotlib.pyplot as plt
import numpy as np
import pylab
import pickle
import random
 

def get_data():

	# mag = random.randint(1,5) # 5 #noise mag varies from 1 to 5
	# dist = random.randint(10,17)/10 #1.7 #distortion mag 1 to 1.7

	Fs = 1000#samples will remain constant
	sample = 1000
	x = np.arange(sample)
	noise = 2 * 0.0001*np.asarray(random.sample(range(0,1000),sample))
	 
	wave = np.sin(2 * np.pi * 1 * x / Fs) + noise + 1 * x/sample

	wave2 = np.concatenate((wave, noise), axis=None)

	return wave2
 


wave = get_data()

test_data = []

for i in range(19):
	a = wave[i*50:i*50 + 1000]
	test_data.append(a)

	
with open("test_data.txt", "wb") as fp:
	pickle.dump(test_data, fp, protocol=2)

# with open("input_data_labels.txt", "wb") as fp:
# 	pickle.dump(true_labels, fp, protocol=2)

# with open('input_data.txt', 'wb') as f:
#     for item in sliding_window:
#         f.write("%s\n" % item)

import pdb;pdb.set_trace()
sample = 2000
x = np.arange(sample)
y = wave
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# import pdb;pdb.set_trace()
# print(sliding_window)