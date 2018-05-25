# Traffic Sign Classifier

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_images/class_histogram.png "Class Histogram"
[image2]: ./write_up_images/grayscaled.png "Preprocessed image"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/bumpy_road.jpg "Traffic Sign 1"
[image5]: ./test/priority_road.jpg "Traffic Sign 2"
[image6]: ./test/Road_Work.jpg "Traffic Sign 3"
[image7]: ./test/stop.jpg "Traffic Sign 4"
[image8]: ./test/yield.jpg "Traffic Sign 5"
[softmax]: ./write_up_images/softmax.png "Softmax Probabilities"



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy shape attribute to output size statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Below is an exploratory visualization of the data set. It is a histogram showing the frequency of the classes in the data set. Adding the class names (labels) clutters the chart too much, so to see the mapping of class number to label, see [here.](./sign_names.csv)

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce computational cost as well as reduce complexity of the model.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this greatly increases the probability that the cost function is well-behaved. When feeding an image into the network, if the image data is not normalized, for one image, a particular weight may be too large, and for another, it may be too small. Since we are multiplying the data by the weights, large data values could cause the gradient updates to go out of control, while small data values could have little effect. The two pictures may have the same feature that we would like to recognize, but the pixel intensities may be significantly different. Normalizing helps to close that gap.

I decided not to generate additional data as a step to keep computational cost down. I planned to do it if I needed an accuracy boost, however, it was shown to not be necessary, as most runs ended up with an accuracy above 93%. If I attempted to increase the accuracy to 99%+, then augmenting the dataset would probably be necessary.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a modified LeNet architecture. It consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x72      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x72 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x96      									|
| RELU					|												|
| Flatten					|		Total neurons = 864					|
| Fully connected		| Input = 864, Output = 240					|
| RELU					|												|
| Fully connected		| Input = 240, Output = 84					|
| RELU					|												|
| Fully connected		| Input = 84, Output = 43 (number of classes)			|
| Softmax				|        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer for 25 epochs, with a batch size of 128 and a learning rate of 0.001.. I spent much more time adjusting the model architecture than adjusting the hyperparameters as it seemed that changes to the architecture had a much more significant impact on the accuracy than changing hyperparameters.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: not measured and is fairly meaningless
* validation set accuracy: 94.4%
* test set accuracy: 93.2%

As discussed previously, the LeNet architecture was chosen. This architecture has been shown to do well on classifying images. I originally chose it because I was familiar with the architecture and knew it would make a good starting point. This turned out to be a good choice, because the architecture performed well with only a few small modifications. Many training runs showed validation set accuracy over 96%. The last epoch of the run currently displayed in the jupyter notebook is 94.4%.

The model's performance on the test set was 93.3%, which is fairly close to the validation set accuracy, implying the model is performing well and hasn't been overfit, despite not taking significant steps to prevent overfitting. I only trained for 25 epochs, Training any longer would likely start to result in overfitting unless I added dropout or augmented the dataset.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I expected the first three images, bumpy road, priority road, and road work, to be fairly easy for the model to classify. They are nearly square images where the sign fills most of the photo, so resizing and preprocessing these three images results in images with very similar style to the training set images.

For the last two images, stop and yield, I expected the model to have a little more trouble. In these two, the image is not square, and the sign does not fill the entire image, so it gets distorted when preprocessed.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road   									| 
| Priority Road     			| Priority Road 										|
| Road Work      | Road Work |
| Stop  | No passing  |
| Yield					| Yield											|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This shows that the model definitely has promise, but a set of 5 isn't exactly statistically significant, so it's difficult (and erroneous) to make strong conclusions from this test. The model had difficulty classifying the stop sign, likely due to the distortion resulting from preprocessing. The model did not have any difficulty classifying the yield sign, however. I believe this is because the yield sign has a unique shape, while many of the signs in the dataset had a round shape, like the stop sign. To become better at classifying the stop sign, I believe augmenting the dataset with distorted (rotated, stretched, blurred, etc.) images would be the best route to take.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For all images except for the stop sign, the model is extremely sure (99.5%+) of its classification. However, for the stop sign, the correct label is not even in the top five guesses. The softmax probabilities are shown in the jupyter notebook and repeated below:

![alt_text][softmax]
