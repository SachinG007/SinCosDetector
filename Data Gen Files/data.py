import matplotlib.pyplot as plt
import numpy as np
import pylab
import pickle
import random
 

def get_data():

	just_noise = 1#random.randint(1,3)
	f = 1
	wave_type = 1#random.randint(1,2) #1 # +1 or -1 , wave A or wave B

	mag = 1#random.randint(1,5) # 5 #noise mag varies from 1 to 5
	dist = 1#random.randint(10,17)/10 #1.7 #distortion mag 1 to 1.7

	Fs = 1000#samples will remain constant
	sample = 1000
	x = np.arange(sample)
	noise = mag * 0.0001*np.asarray(random.sample(range(0,1000),sample))
	 
	wave = np.sin(2 * np.pi * f * x / Fs) + noise + dist * x/sample

	if wave_type == 2:
		wave = -1 * wave

	if just_noise == 3:
		wave = noise
		wave_type = 0

	# plt.plot(x, y)
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.show()
	wave = (wave - np.mean(wave))/np.std(wave)
	return wave,wave_type
 

sliding_window = []
true_labels = []
 
for i in range(1):
	wave, wave_type = get_data()
	sliding_window.append(wave)
	true_labels.append(wave_type)

# with open("input_data.txt", "wb") as fp:
# 	pickle.dump(sliding_window, fp, protocol=2)

# with open("input_data_labels.txt", "wb") as fp:
# 	pickle.dump(true_labels, fp, protocol=2)

# with open('input_data.txt', 'wb') as f:
#     for item in sliding_window:
#         f.write("%s\n" % item)

sample = 1000
x = np.arange(sample)
y = sliding_window[0]
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# import pdb;pdb.set_trace()
# print(sliding_window)