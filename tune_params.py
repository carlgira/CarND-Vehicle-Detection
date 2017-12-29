import vehicle_detection as vd
import random
import time
from pprint import pprint

def param_generator():
	color_space_values = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb' ]
	orient_values = [5, 7, 9, 12, 14]
	pix_per_cell_values = [6, 7, 8, 9, 10]
	cell_per_block_values = [2, 3, 4]
	hog_channel_values = [0, 1, 2, 'ALL']
	spatial_size_values = [(16,16), (24, 24), (32, 32), (40, 40), (48, 48)]
	hist_bins_values = [24, 28, 32, 36, 40]
	spatial_feat_values = True
	hist_feat_values = True
	hog_feat_values = True
	rand_state = 41

	param = vd.Params(random.choice(color_space_values), random.choice(orient_values), random.choice(pix_per_cell_values),
					  random.choice(cell_per_block_values), random.choice(hog_channel_values), random.choice(spatial_size_values),
					  random.choice(hist_bins_values), spatial_feat_values, hist_feat_values,
					  hog_feat_values, rand_state)

	return param


class LinearSVC_Exec:
	def __init__(self):
		self.param = param_generator()
		self.time = 0
		self.accuracy = 0

	def test_params(self):
		t=time.time()
		_, _, self.accuracy = vd.train(self.param)
		t2 = time.time()
		self.time = round(t2-t, 2)

	def __str__(self):
		return str(str(self.param) + ", time: " + str(self.time) + ", accuracy: " + str(self.accuracy))



def tune_params(population):

	for p in range(population):
		try:
			lsvc_exec = LinearSVC_Exec()
			lsvc_exec.test_params()
			print(lsvc_exec)
		except:
			continue

if __name__ == '__main__':
	tune_params(100)



