import vehicle_detection as vd
import random
import time
import pandas as pd

def param_generator():
	color_space_values = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb' ]
	orient_values = [5, 7, 9, 12, 14]
	pix_per_cell_values = [6, 7, 8, 9, 10]
	cell_per_block_values = [2, 3, 4]
	hog_channel_values = [0, 1, 2, 'ALL']
	spatial_size_values = [(16,16), (24, 24), (32, 32), (40, 40), (48, 48)]
	hist_bins_values = [24, 28, 32, 36, 40, 48]
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

	def get_pd(self):
		return {'color_space': self.param.color_space, 'orient': self.param.orient, 'pix_per_cell' : self.param.pix_per_cell,
				'cell_per_block' : self.param.cell_per_block, 'hog_channel': self.param.hog_channel,
				'spatial_size' : self.param.spatial_size, 'hist_bins': self.param.hist_bins, 'time' : self.time, 'accuracy': self.accuracy }

	def __str__(self):
		return str(str(self.param) + ", time: " + str(self.time) + ", accuracy: " + str(self.accuracy))



def tune_params(population):

	list_sols = pd.DataFrame([], columns=['color_space', 'orient', 'pix_per_cell', 'cell_per_block', 'hog_channel',
										  'spatial_size', 'hist_bins', 'time', 'accuracy'])
	for p in range(population):
		try:
			lsvc_exec = LinearSVC_Exec()
			lsvc_exec.test_params()
			list_sols = list_sols.append(lsvc_exec.get_pd(), ignore_index=True)
			print(lsvc_exec)
		except:
			continue

	list_sols.sort_values('accuracy', inplace=True, ascending=False)
	list_sols.to_csv('lsvc_conf.csv')
	print(list_sols)

	return list_sols


if __name__ == '__main__':
	tune_params(100)



