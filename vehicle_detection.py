from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
#from lane_detection import process_image_vid as draw_lane_lines
import os
from moviepy.editor import VideoFileClip
import pickle

def convert_color(img, conv='RGB2YCrCb'):
	if conv == 'RGB2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	if conv == 'BGR2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	if conv == 'RGB2LUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis:
		features, hog_image = hog(img, orientations=orient,
								  pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block),
								  transform_sqrt=True,
								  visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	else:
		features = hog(img, orientations=orient,
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block),
					   transform_sqrt=True,
					   visualise=vis, feature_vector=feature_vec)
		return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel()
	# Return the feature vector
	return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
					 hist_bins=32, orient=9,
					 pix_per_cell=8, cell_per_block=2, hog_channel=0,
					 spatial_feat=True, hist_feat=True, hog_feat=True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		file_features = []
		# Read in each one by one
		image = mpimg.imread(file)
		# apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else:
			feature_image = np.copy(image)

		if spatial_feat:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)

		if hist_feat:
			# Apply color_hist()
			hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(hist_features)

		if hog_feat:
			# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:, :, channel],
														 orient, pix_per_cell, cell_per_block,
														 vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
												pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			# Append the new feature vector to the features list
			file_features.append(hog_features)
		features.append(np.concatenate(file_features))

	# Return list of feature vectors
	return features


notcars = glob.glob('data/non-vehicles/**/*.png')
cars = glob.glob('data/vehicles/**/*.png')

#notcars = notcars[0:1000]
#cars = cars[0:1000]

class Params:
	def __init__(self, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
				 spatial_size=(32,32), hist_bins=32, spatial_feat=True, hist_feat=True, hog_feat=True, rand_state=None):
		self.color_space = color_space
		self.orient = orient
		self.pix_per_cell = pix_per_cell
		self.cell_per_block = cell_per_block
		self.hog_channel = hog_channel
		self.spatial_size = spatial_size
		self.hist_bins = hist_bins
		self.spatial_feat = spatial_feat
		self.hist_feat = hist_feat
		self.hog_feat = hog_feat
		self.rand_state = rand_state

	def __str__(self):
		return str("color_space: " + self.color_space +  ", orient: " + str(self.orient) +  ", pix_per_cell: " + str(self.pix_per_cell) + ", cell_per_block: " + str(self.cell_per_block) + ", hog_channel: " + str(self.hog_channel) + \
			   ", spatial_size: " + str(self.spatial_size) + ", hist_bins: " + str(self.hist_bins) + ", spatial_feat: " + str(self.spatial_feat) + ", hist_feat: " + str(self.hist_feat) + ", hog_feat :" + str(self.hog_feat) + ", rand_state: " + str(self.rand_state))


ystart = 400
ystop = 656
scale = 1.5


def train(params):

	car_features = extract_features(cars, color_space=params.color_space,
									spatial_size=params.spatial_size, hist_bins=params.hist_bins,
									orient=params.orient, pix_per_cell=params.pix_per_cell,
									cell_per_block=params.cell_per_block,
									hog_channel=params.hog_channel, spatial_feat=params.spatial_feat,
									hist_feat=params.hist_feat, hog_feat=params.hog_feat)
	notcar_features = extract_features(notcars, color_space=params.color_space,
									   spatial_size=params.spatial_size, hist_bins=params.hist_bins,
									   orient=params.orient, pix_per_cell=params.pix_per_cell,
									   cell_per_block=params.cell_per_block,
									   hog_channel=params.hog_channel, spatial_feat=params.spatial_feat,
									   hist_feat=params.hist_feat, hog_feat=params.hog_feat)

	X = np.vstack((car_features, notcar_features)).astype(np.float64)

	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# Split up data into randomized training and test sets
	if params.rand_state is None:
		rand_state = np.random.randint(0, 100)
	else:
		rand_state = params.rand_state

	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.2, random_state=rand_state)

	svc_model_file = 'svc_pickle.p'
	if os.path.exists(svc_model_file):
		dist_pickle = pickle.load(open(svc_model_file, "rb"))
		svc = dist_pickle["svc"]
		X_scaler = dist_pickle["scaler"]
	else:
		svc = LinearSVC()
		svc.fit(X_train, y_train)
		with open(svc_model_file, 'wb') as handle:
			pickle.dump({"svc": svc, "scaler": X_scaler}, handle, protocol=pickle.HIGHEST_PROTOCOL)


	# Check the score of the SVC
	accuracy = round(svc.score(X_test, y_test), 4)
	print('Test Accuracy of SVC = ', accuracy)
	# Check the prediction time for a single sample
	return X_scaler, svc, accuracy



# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, draw_all_boxes=False):
	draw_img = np.copy(img)
	img = img.astype(np.float32) / 255

	img_tosearch = img[ystart:ystop, :, :]
	ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

	ch1 = ctrans_tosearch[:, :, 0]
	ch2 = ctrans_tosearch[:, :, 1]
	ch3 = ctrans_tosearch[:, :, 2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
	nfeat_per_block = orient * cell_per_block ** 2

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	bbox_list = []
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb * cells_per_step
			xpos = xb * cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
			hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
			hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos * pix_per_cell
			ytop = ypos * pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(
				np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
			# test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
			test_prediction = svc.predict(test_features)

			if draw_all_boxes:
				xbox_left = np.int(xleft * scale)
				ytop_draw = np.int(ytop * scale)
				win_draw = np.int(window * scale)
				bbox_list.append(
					((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
				cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
							  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (255, 0, 0), 6)

			if test_prediction == 1:
				xbox_left = np.int(xleft * scale)
				ytop_draw = np.int(ytop * scale)
				win_draw = np.int(window * scale)
				bbox_list.append(
					((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
				cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
							  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

	return draw_img, bbox_list


def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap


def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1] + 1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
	# Return the image
	return img


def draw_real_boxes(image, box_list):
	heat = np.zeros_like(image[:, :, 0]).astype(np.float)

	# Add heat to each box in box list
	heat = add_heat(heat, box_list)

	# Apply threshold to help remove false positives
	heat = apply_threshold(heat, 1)

	# Visualize the heatmap when displaying
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(image), labels)

	return draw_img

def imgage_process(image, params):

	svc_model_file = 'svc_pickle.p'
	if os.path.exists(svc_model_file):
		dist_pickle = pickle.load(open(svc_model_file, "rb"))
		svc = dist_pickle["svc"]
		X_scaler = dist_pickle["scaler"]
	else:
		X_scaler, svc = train(params, cars, notcars)
		with open(svc_model_file, 'wb') as handle:
			pickle.dump({"svc": svc, "scaler": X_scaler}, handle, protocol=pickle.HIGHEST_PROTOCOL)


	box_img, box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, params.orient, params.pix_per_cell, params.cell_per_block, params.spatial_size, params.hist_bins)

	#draw_img = draw_lane_lines(image)

	draw_img = draw_real_boxes(image, box_list)

	draw_img_scaled = cv2.resize(draw_img, (0,0), fx=0.5, fy=0.5)
	box_img_scaled = cv2.resize(box_img, (0,0), fx=0.5, fy=0.5)

	final = np.zeros((int(image.shape[0]/2), int(image.shape[1]), 3), dtype=np.uint8)
	final[:,0:int(final.shape[1]/2)] = draw_img_scaled
	final[:,int(final.shape[1]/2):final.shape[1]] = box_img_scaled

	return final

def process_video(video, video_output):
	""" Process frames of video using the process image function to draw lane lines"""
	clip1 = VideoFileClip(video)
	clip = clip1.fl_image(imgage_process)
	clip.write_videofile(video_output, audio=False, verbose=False, progress_bar=False)

if __name__ == '__main__':
	image = mpimg.imread('test_images/test1.jpg')
	params = Params(color_space='YCrCb', orient=12, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
					spatial_size=(32,32), hist_bins=48,spatial_feat=True, hist_feat=True, hog_feat=True)

	p_image = imgage_process(image, params)
	#plt.imshow(p_image)
	#plt.show()

	#process_video("test_video.mp4", "output_videos/test_video.mp4")
