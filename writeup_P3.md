# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 - A video recording of vehicle driving autonomously for one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is built based Nvidia model. It works good for this task. First, I have to pre-process images by cropping them, to avoid sky and hood of the car.
The code can be found in `Nvidia_model` function. 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

I also flipped the images to get more traning data 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to let the car drive completely on the track.

My first step was to use a convolution neural network model in the first lecture with just dense and flatten, just to test it out. It works not good(of course).
 
Then I tried other models like LeNet. And I started to record my own training data with as many situations as possible.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model was overfitting. 

To combat the overfitting, I modified the model by adding several dropout.

Then I found it had a high loss even I set the model properly. So I changed it to Nvidia model. It works better. So I finally chose it for this project.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at first few tries and it ran slowly.
To improve the driving behavior in these cases, I modified the model and improved my train set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py `Nvidia_model` function) consisted of a convolution neural network with the following layers and layer sizes ...

* Convolutional layer, 24 of 5x5, subsample 2x2, activation relu
* Dropout Layer, probability 0.5
* Convolutional layer, 36 of 5x5, subsample 2x2, activation relu
* Dropout Layer, probability 0.5
* Convolutional layer, 48 of 5x5, subsample 2x2, activation relu
* Dropout Layer, probability 0.5
* Convolutional layer, 64 of 5x5, subsample 1x1, activation relu
* Dropout Layer, probability 0.5
* Convolutional layer, 64 of 5x5, subsample 1x1, activation relu
* Dropout Layer, probability 0.5
* Flatten
* Fully-Connected, 100 Neurons
* Fully-Connected, 50 Neurons
* Fully-Connected, 10 Neurons
* Fully-Connected, 1 Neuron

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used all three center, left and right camera data recorded from one lap on track. For left and right camera, I applied a correction of 0.2 so that the vehicle would learn to handle different situations correctly.  
To augment the data sat, I also flipped images and angles thinking that this would add more data into data set, and it has more training for right turn as well.
I believe all those can make result more accurate.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

Here is the output of the training:

```
   1520/1520 [==============================] - 587s - loss: 0.0344 - val_loss: 0.0432
   Epoch 2/5
   1520/1520 [==============================] - 582s - loss: 0.0258 - val_loss: 0.0334
   Epoch 3/5
   1520/1520 [==============================] - 477s - loss: 0.0215 - val_loss: 0.0271
   Epoch 4/5
   1520/1520 [==============================] - 580s - loss: 0.0190 - val_loss: 0.0258
   Epoch 5/5
   1520/1520 [==============================] - 580s - loss: 0.0175 - val_loss: 0.0265
```

The ideal number of epochs was 5 since there were up and downs after that. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### P.S.
As you may see from the video of my auto run result, the car always stays on the track, which is good. But it's sometimes close to the edge.
 
I think this probably be due to the training data I provided. Because I played too much car racing games, I always try to find the best route on the track when I was driving. This should be something to improve.
