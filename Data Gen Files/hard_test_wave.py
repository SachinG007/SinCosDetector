import matplotlib.pyplot as plt
import numpy as np
import pylab
import pickle
import random
 

def get_data():

	just_noise = random.randint(1,2) #noise or not
	pure_noise = random.randint(1,3) #if noise , pure noise or wavy noise	
	f = 1#random.randint(1,3) #1 #num of cycles
	wave_type = 2#random.randint(1,2) #1 # +1 or -1 , wave A or wave B

	mag = 1#random.randint(1,2) # 5 #noise mag varies from 1 to 5
	dist = .10#random.randint(10,20)/100 #1.7 #distortion mag 1 to 1.7


	Fs = 50#samples will remain constant
	sample = 50
	x = np.arange(sample)
	noise = mag * 0.0001*np.asarray(random.sample(range(0,1000),sample))
	wave_c = -1 * np.sin(2 * np.pi * f * x / Fs ) + noise + dist * x/sample

	# noise = mag * 0.0001*np.asarray(random.sample(range(0,1000),25))
	# wave_c = np.concatenate((wave, wave), axis=None)

	# wave = np.sin(2 * np.pi * f * x / Fs ) + noise + dist * x/sample

	# wave_c = np.concatenate((wave_c, wave), axis=None)
	# plt.plot(x, y)
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.show()
	# wave = (wave - np.mean(wave))/np.std(wave)
	return wave_c,wave_type
 

sliding_window = []
true_labels = []
 

# wave, wave_type = get_data()
# for i in range(11):
# 	a = wave[i*5:i*5 + 50]
# 	sliding_window.append(a)
# 	if i<2:
# 		true_labels.append(1)
# 	elif i==9:
# 		true_labels.append(1)
# 	else:
# 		true_labels.append(0)

for i in range(256):
	wave, wave_type = get_data()
	print(wave_type)
	sliding_window.append(wave)
	true_labels.append(wave_type)

# sliding_window.append(wave)
# true_labels.append(wave_type)

with open("htest_data.txt", "wb") as fp:
	pickle.dump(sliding_window, fp, protocol=2)

with open("htest_data_labels.txt", "wb") as fp:
	pickle.dump(true_labels, fp, protocol=2)

# with open('input_data.txt', 'wb') as f:
#     for item in sliding_window:
#         f.write("%s\n" % item)

sample = 50
x = np.arange(sample)
y = sliding_window[7]
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# print(true_labels[0])
# import pdb;pdb.set_trace()
# print(sliding_window)