import csv
import cv2
import numpy as np
import ntpath
import time
import pickle
import sklearn
import tensorflow
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

# Read csv data from file
def read_csv(file_name):
    samples = []
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(line)
    return samples
    
# Use generator to use less memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = [] # Processed images
            angles = [] # Strering angles
            for batch_sample in batch_samples:
                # Use all imgages captured by center, left and right camera
                path = './train_data'
                # path = '/opt/carnd_p3/data'
                img_center = cv2.cvtColor(cv2.imread(path + '/IMG/' + batch_sample[0].split('/')[-1]), cv2.COLOR_BGR2RGB)
                img_left = cv2.cvtColor(cv2.imread(path + '/IMG/' + batch_sample[1].split('/')[-1]), cv2.COLOR_BGR2RGB)
                img_right = cv2.cvtColor(cv2.imread(path + '/IMG/' + batch_sample[2].split('/')[-1]), cv2.COLOR_BGR2RGB)

                images.append(img_center)
                images.append(img_left)
                images.append(img_right)

                # Same as images, use all stering angles of center
                # Left and right angles are based on center but applied a correction to them
                steering_center = float(batch_sample[3])
                steering_left = steering_center + 0.2
                steering_right = steering_center - 0.2

                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)
                
                # Data augmentation
                images.append(np.fliplr(img_center))
                images.append(np.fliplr(img_left))
                images.append(np.fliplr(img_right))

                angles.append(steering_center * -1.0)
                angles.append(steering_left * -1.0)
                angles.append(steering_right * -1.0)
                
            # Make final train data
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    
# Process pipeline    

# Local file
file_local = './train_data/driving_log.csv'

# Data provided by course.
file_given = '/opt/carnd_p3/data/driving_log.csv'

# Read csv file first, get images and angles data
samples = read_csv(file_local)

# Splite raw data into train and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Create train and validation data using generater
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Build NVidia model
def Nvidia_model():
    model = Sequential()   
    # Pre-process data first, crop them to remove sky and hood part
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0)))) 

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.5))    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# Use adam optimizer
model = Nvidia_model()
model.compile(loss='mse', optimizer='adam')
# Fit model
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps = len(validation_samples), epochs=5, verbose = 1)

# Save model file
model.save('model.h5')
