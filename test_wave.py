import matplotlib.pyplot as plt
import numpy as np
import pylab
import pickle
import random
 

def get_data():

	just_noise = random.randint(1,2) #noise or not
	pure_noise = random.randint(1,3) #if noise , pure noise or wavy noise	
	f = 1#random.randint(1,3) #1 #num of cycles
	noise_shift = random.randint(10,20)
	noise_shift = noise_shift/5
	wave_type = random.randint(1,2) #1 # +1 or -1 , wave A or wave B

	mag = 1#random.randint(3,4) # 5 #noise mag varies from 1 to 5
	dist = random.randint(10,20)/15 #1.7 #distortion mag 1 to 1.7
	# dist = random.randint(30,40)/100 #1.7 #distortion mag 1 to 1.7

	Fs = 50#samples will remain constant
	sample = 50
	x = np.arange(sample)
	noise = mag * 0.0001*np.asarray(random.sample(range(0,1000),sample))
	wave = np.sin(2 * np.pi * f * x / Fs ) + noise + dist * x/sample

	if wave_type == 2:
		wave = -1 * wave

	if just_noise == 2:
		if pure_noise==3:

			wave = noise
			wave_type = 0

		else:
			wave = np.sin(2 * np.pi * f * x / Fs + np.pi/noise_shift) + noise + dist * x/sample	
			disp = int(sample/(2*noise_shift))		
			wave[sample-disp:sample] = mag * 0.0001*np.asarray(random.sample(range(0,1000),disp))
			if wave_type == 2:
				wave = -1 * wave
			wave_type = 0



	# plt.plot(x, y)
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.show()
	wave = (wave - np.mean(wave))/np.std(wave)
	return wave,wave_type
 

sliding_window = []
true_labels = []
 
for i in range(256):
	wave, wave_type = get_data()
	print(wave_type)
	sliding_window.append(wave)
	true_labels.append(wave_type)

#Bakchodi
# wave_c = wave_c = np.concatenate((sliding_window[0],sliding_window[1]), axis=None)

# sliding_window = []
# true_labels = []
# for i in range(11):

# 	a = wave_c[i*5:i*5 + 50]
# 	sliding_window.append(a)
# 	if i<2:
# 		true_labels.append(1)
# 	elif i==10:
# 		true_labels.append(1)
# 	else:
# 		true_labels.append(0)


with open("test_data.txt", "wb") as fp:
	pickle.dump(sliding_window, fp, protocol=2)

with open("test_data_labels.txt", "wb") as fp:
	pickle.dump(true_labels, fp, protocol=2)

# with open('input_data.txt', 'wb') as f:
#     for item in sliding_window:
#         f.write("%s\n" % item)

sample = 50
x = np.arange(sample)
y = sliding_window[10-1]
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# print(true_labels[0])
# import pdb;pdb.set_trace()
# print(sliding_window)