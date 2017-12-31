# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

**Files in project**

- **Readme.md :** Writeup of project (this file)
- **Vehicle-Detection.ipynb :** Notebook with all the code and notes.
- **lsvc_hog_params.csv:** List of tested parameters the Linear  SVM
- **svc_pickle.p:** Saved LinearSVC model
- **output_videos:** Output videos

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/bboxes_and_heat1.png
[image9]: ./examples/bboxes_and_heat2.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained the section **"Histogram of Oriented Gradients (HOG)"**.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

For deciding the parameters i build a random generator creating 100 possible combinations for the LinearSVC. I train the model using a subset of 4000 images. You can see the random generator in the section **Tunning HOG parameters**

I show the first 20 combinations:

colorSpace|orient|pix_per_cell|cell_per_block|hogChannel|spatialSize|histBins|time |accuracy
----------|------|------------|--------------|----------|-----------|--------|-----|--------
YCrCb      |7.0   |8.0         |2.0           |ALL        |(32, 32)    |36.0     |12.84|1.0
YCrCb      |9.0   |8.0         |3.0           |ALL        |(16, 16)    |48.0     |12.65|0.9975
YUV        |9.0   |8.0         |4.0           |ALL        |(48, 48)    |24.0     |14.04|0.9975
HSV        |9.0   |6.0         |3.0           |ALL        |(24, 24)    |28.0     |14.65|0.9975
YCrCb      |12.0  |9.0         |4.0           |ALL        |(24, 24)    |32.0     |11.02|0.995
HSV        |12.0  |10.0        |2.0           |ALL        |(16, 16)    |24.0     |9.39 |0.995
HSV        |14.0  |6.0         |3.0           |2          |(24, 24)    |36.0     |7.22 |0.995
YCrCb      |5.0   |9.0         |4.0           |ALL        |(24, 24)    |48.0     |10.86|0.995
YUV        |5.0   |8.0         |3.0           |ALL        |(32, 32)    |32.0     |12.74|0.995
HLS        |14.0  |7.0         |3.0           |1          |(48, 48)    |48.0     |7.55 |0.9925
YUV        |12.0  |10.0        |3.0           |ALL        |(16, 16)    |36.0     |9.36 |0.9925
RGB        |9.0   |7.0         |2.0           |ALL        |(16, 16)    |32.0     |12.18|0.9925
RGB        |14.0  |8.0         |3.0           |2          |(40, 40)    |48.0     |7.96 |0.9925
RGB        |12.0  |7.0         |3.0           |1          |(24, 24)    |48.0     |6.24 |0.9925
YUV        |9.0   |6.0         |4.0           |ALL        |(48, 48)    |24.0     |23.8 |0.9925
LUV        |14.0  |8.0         |3.0           |0          |(40, 40)    |40.0     |9.42 |0.9925
RGB        |12.0  |8.0         |3.0           |ALL        |(24, 24)    |36.0     |12.76|0.99
YCrCb      |14.0  |6.0         |4.0           |ALL        |(24, 24)    |48.0     |22.27|0.99
HSV        |12.0  |7.0         |2.0           |ALL        |(16, 16)    |32.0     |15.32|0.99
RGB        |14.0  |7.0         |4.0           |1.0        |(48, 48)    |28.0     |9.69 |0.99
YCrCb      |12.0  |6.0         |3.0           |ALL        |(32, 32)    |40.0     |18.0 |0.99
LUV        |9.0   |8.0         |4.0           |0          |(32, 32)    |32.0     |6.21 |0.99
RGB        |7.0   |9.0         |2.0           |ALL        |(40, 40)    |48.0     |12.37|0.99
YUV        |12.0  |8.0         |4.0           |0          |(32, 32)    |40.0     |6.65 |0.99
YCrCb      |14.0  |10.0        |3.0           |0          |(16, 16)    |32.0     |5.31 |0.9875

#### 2. Explain how you settled on your final choice of HOG parameters.

- Always use the three types of features activating the flags spatial_feat_values, hist_feat_values,hog_feat_values. I dit not include the flags in the table because the results where bad without one or two of the features. (i did some previous calculation)
- Is clear that is useful to use **ALL** for the channels of the HOG.
- For the **color space** there is a tendency to use the **YCrCb, or RGB**.
- For the **orient** variable is seems to be better to use a higher value than the default, something around **12**
- A higger value on the numbers of bins seems to work better, set to **48**.
- The pix_per_cell, cell_per_block and the spatial_size seems to work well with the default values.

The final configuration is this:

```python
params = Params(color_space='YCrCb', 
                orient=12, pix_per_cell=8, 
                cell_per_block=2, 
                hog_channel='ALL',
                spatial_size=(32,32), 
                hist_bins=48, 
                spatial_feat=True, 
                hist_feat=True, 
                hog_feat=True)
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I train a Linear SVM, usign the hog and color features. I tune the parameters used for the hog features doing some test runs.

After tuning the parameters the accuracy of the model reach **0.9994**

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

You can see the code in the section titled "Sliding Window Search". The method combines HOG and color feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The overlaping of bounding boxes was on 50%.

The method performs the classifier prediction and returns a list of boxes corresponding with the positions of the car predictions.

The image below shows the first attempt at using find_cars on one of the test images, using a single window size:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using a linear SVM using the hog and color features as input. The tuning of the model was done previously finding optimus parametes for the feature extraction.

The resulting rectagles after passign the linear SVM looks like this.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most part of the implemention of the functions were on the course itself, so the main problem of the project reside on the tunning of the parameters. Thats why i build a random generator of parameters to specially tune the values used for the linear SVM. After making some runs and getting 100 different combinations of parameters i was able to choose ones that help me get an accuracy on the test set of 0.9994.

The model behaves well with the project video, but i doubt that it will generalize for other cases. The model will probably fail with other lighting conditions or simply usign a type of car that is not in the dataset (a truck). 

The pipeline is able to detect and eliminate most of the false positives, there are still one or two errors in the video.

For making a more robust model it would be necesary lots of more data, with other types of vehicles in other positions and different light conditions, and of course use state of the art neural network architecture for this task.
