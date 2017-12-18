# **Behavioral Cloning**

## Project writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_docs/flipped.png "Flipped images"
[image2]: ./writeup_docs/cropped.png "Cropped image"
[image3]: ./writeup_docs/model_performance.png "Model Performance"
[image4]: ./writeup_docs/distribution_before_augmenting.png "Distribution before augmenting"
[image5]: ./writeup_docs/distribution_after_augmenting.png "Distribution after augmenting"
[image6]: ./writeup_docs/left.jpg "Normal Image (Left)"
[image7]: ./writeup_docs/right.jpg "Normal Image (Right)"
[image8]: ./writeup_docs/center.jpg "Normal Image (Center)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model. All the paths are defined in the CONSTANTS variable
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained neural network which simulates driving
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network
It also contains the pipeline I used for training and validating the model.

There are 2 approaches available within the
pipeline: the first one uses generators in order to reduce the amount of RAM used for training and the second one
allows the whole dataset to be stored in the RAM (my PC has enough memory, though I've used mainly this approach because
it seems to be faster for about ~20% than fitting model using generators).

All the functionalty used within the pipeline is divided in functions, so they could be easily exported and reused in
other applications.

### Model Architecture and Training Strategy

#### 1. Architecture overview

As a basic architecture **Nvidia's** one was used. I've played a bit with activations in dense layers and found out that it
is better to use RELUs rather than ELUs for convolutional layers - it allows to increase performance of the model for
percents. Also I've found that using sigmoid activation in the very last layer allows to smoothify the way the car
drives around the track, so it behaves in a more natural way.

Lambda layer is used in order to crop the image so that only the road is visible on the image. That measure allows
to prevent the model from being distracted by the visible objects which shouldn't affect steering angle.


Cropped image fed to network:

![alt text][image2]

Resulting architecture of the network is following:

Layer (type)               |  Output Shape           |   Param #
---|---|---
cropping2d_1 (Cropping2D)  |  (None, 70, 320, 3)     |   0
lambda_1 (Lambda)          |  (None, 70, 320, 3)     |   0
conv2d_1 (Conv2D, Activation - RELU)          |  (None, 33, 158, 24)    |   1824
conv2d_2 (Conv2D, Activation - RELU)          |  (None, 15, 77, 36)     |   21636
conv2d_3 (Conv2D, Activation - RELU)          |  (None, 6, 37, 48)      |   43248
conv2d_4 (Conv2D, Activation - RELU)          |  (None, 4, 35, 64)      |   27712
conv2d_5 (Conv2D, Activation - RELU)          |  (None, 2, 33, 64)      |   36928
dropout_1 (Dropout)        |  (None, 2, 33, 64)      |   0
flatten_1 (Flatten)        |  (None, 4224)           |   0
dense_1 (Dense)            |  (None, 100)            |   422500
dense_2 (Dense)            |  (None, 50)             |   5050
dense_3 (Dense, Activation - RELU)            |  (None, 10)             |   510
dense_4 (Dense, Activation - Sigmoid)            |  (None, 1)              |   11


>Total params: 559,419

>Trainable params: 559,419

>Non-trainable params: 0


#### 2. Data collection strategies and attempts to reduce overfitting in the model

To create appropriate training dataset two records were made:
1. Driving two laps across the track
2. Driving two laps in the opposite direction

Mouse had been used as a controller in order to provide smooth and natural driving behavior. Training data didn't
include any extremal cases like sharp turns or external brakes. The model should drive in a safe way, so the example we provide
should follow the same principle :)

Initial dataset was unbalanced - the number of images with angle values close to zero was huge and first attempts to train the model failed because model was taught to drive as straight as possible, so I had to remove part of zero like samples from the dataset. Afterwards the distribution of angles became look more like a normal distribution:

![alt text][image4]


In order to reduce overfitting there is model a dropout layers which is placed after convolutions (model.py line 155, dropout coef=0.5)

Also in order to avoid overfitting and provide more data for the model augmentation had been used. The first augmentation
techique used is flipping image horizontally with setting negative angle as a target value. It is kinda primitive approach,but it has several pros:
1. It allows to use as two times as larger dataset
2. It allows to avoid networks left/right side bias when choosing the direction to drive

Here you can see an example of original and flipped images with their target angle values:

![alt text][image1]

The second technique is about using additional images from the cameras placed on the right/left side of the car's windshield. Things worse mentioning here are:
1. Angle correction should be added (I've used **2.25**)
2. The very bottom of image with the windshield should be cropped (because it makes model to think about the windshield as a crucial feature)

So instead of one input image we have three:

![alt text][image6]
![alt text][image8]
![alt text][image7]

After augmenting the distribution of dataset looks like:

![alt text][image5]

The model was trained and validated on different data sets to ensure that the model was not overfitting.
All the data is splitted randomly into training and validation subsets:
80% ~ training data;
20% ~ validation data

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
The model successfully completed this task as you could see in _video.py_

#### 3. Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually (model.py line 163).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try to teach the car drive the track using previously learned
network architectures and find the most appropriate soultion experimenting with parameters, layers, training dataset and augmentation techniques.

My first step was to use a convolution neural network model similar to the one used in SignClassifier project,
I thought this model might be appropriate because it also works with images as inputs and produces numerical output.
But that model didn't perform well. The measure of performance is MSE for training and validation. For LeNet architecture
the maximum performance I could reach was about 70%.

So I've tried to use another architecture, and 've chosen the one described in [Nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)


After selecting the architecture I've started to play with parameters of the network trying to get better accuracy in driving.
Several approaches were tried:
1. Increasing the number of channels in convolutional layers ~ didn't made any effect
2. Removing layers ~ car drives well when road is approximately straight, but fails when it has to turn
3. Changing activations ~ works fine. The most significant performance increasing happened when I've added sigmoid to the last layer;
the most significant performance degradation - when replaced convolutionals' RELUs with ELUs
4. Playing with batch size ~ doesn't made any effect
5. Changing the number of epochs ~ I've found that it is useless to train more than 10 epochs - overfitting makes loss increasing.
For me the best number of epochs that works is 5 - it is enough to train, but not so much that model begins overfitting

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here is a log of network's training:
```
Train on 18926 samples, validate on 4732 samples
Epoch 1/5
- 20s - loss: 0.0211 - val_loss: 0.0107
Epoch 2/5
- 18s - loss: 0.0098 - val_loss: 0.0078
Epoch 3/5
- 18s - loss: 0.0066 - val_loss: 0.0021
Epoch 4/5
- 18s - loss: 0.0013 - val_loss: 9.2001e-04
Epoch 5/5
- 19s - loss: 8.4005e-04 - val_loss: 7.3946e-04
model saved : "./models/Mon Dec 18 02-16-52 2017/model.h5"
```

And a chart representing network's performance across epochs:

![alt text][image3]

