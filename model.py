#%%
import os
import time
import csv
import numpy as np
import cv2
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Cropping2D, Flatten, Lambda, Conv2D, Dropout
from sklearn.model_selection import train_test_split
#%%
CONSTANTS = {}
CONSTANTS['data_csv'] = './training_data/9/driving_log.csv'
CONSTANTS['images_folder'] = './training_data/9/IMG/'
CONSTANTS['models_folder'] = './models/'
#%%
def load_data():
    data = []
    with open(CONSTANTS['data_csv']) as file:
        content = csv.reader(file)
        for line in content:
            data.append(line)

    return np.array(data)

def get_augmented_data(samples_data):
    augmented_data = []
    for sample_data in samples_data:
        image_path, angle = sample_data

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        flipped_image = np.fliplr(image)
        flipped_angle = -angle

        augmented_data.append((image, angle))
        augmented_data.append((flipped_image, flipped_angle))
    return augmented_data

def model_generator(data, batch_size=32):
    samples_total = len(data)
    while True:
        sklearn.utils.shuffle(data)

        for offset in range(0, samples_total, batch_size):
            samples = data[offset:offset+batch_size]

            images = []
            angles = []

            for sample in samples:
                img_c_path, img_l_path, img_r_path, angle, _throttle, _break, _speed = sample

                angle_correction = 0.225
                angle = float(angle)
                angle_left = angle + angle_correction
                angle_right = angle - angle_correction

                raw_data = [
                    (img_c_path, angle),
                    (img_l_path, angle_left),
                    (img_r_path, angle_right)
                ]

                augmented_data = get_augmented_data(raw_data)

                for container in augmented_data:
                    img, ang = container
                    images.append(img)
                    angles.append(ang)

            yield sklearn.utils.shuffle(np.array(images), np.array(angles))

# if we don't use generator
def get_model_data(data):
    x = []
    y = []
    for line in data:
        img_c_path, img_l_path, img_r_path, angle, _throttle, _break, _speed = line

        angle_correction = 0.225
        angle = float(angle)
        angle_left = angle + angle_correction
        angle_right = angle - angle_correction

        raw_data = [
            (img_c_path, angle),
            (img_l_path, angle_left),
            (img_r_path, angle_right)
        ]

        augmented_data = get_augmented_data(raw_data)

        for container in augmented_data:
            img, ang = container
            x.append(img)
            y.append(ang)

    return sklearn.utils.shuffle(np.array(x), np.array(y))

def save_model(model):
    folder = CONSTANTS['models_folder'] + str(time.asctime().replace(':', '-'))
    file = folder + '/model.h5'
    os.mkdir(folder)
    model.save(file)
    print('model saved : \"{}\"'.format(file))

def get_image_size(data):
    image_path = data[0][0]
    image = cv2.imread(image_path)

    return image.shape

def show_initial_angles_distribution(data):
    angles = []
    for line in data:
        angle = float(line[3])
        angles.append(angle)
    angles = np.array(angles)

    plt.title('Initial')
    plt.hist(angles, bins=100, range=(-1,1))
    plt.show()

def show_model_angles_distribution(data):
    plt.title('Result')
    plt.hist(data, bins=100, range=(-1,1))
    plt.show()

def show_model_performance(history_object):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model performance')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
#%%
data = load_data()
image_shape = get_image_size(data)
show_initial_angles_distribution(data)
#%%
with tf.device('/gpu:0'):
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=image_shape))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    use_generator = False

    if use_generator:
        training_data, validation_data = train_test_split(data, test_size=0.2)

        training_generator = model_generator(training_data)
        validation_generator = model_generator(validation_data)

        history_object = model.fit_generator(
            training_generator,
            steps_per_epoch=len(training_data),
            epochs=5,
            validation_data=validation_generator,
            validation_steps=len(validation_data),
            verbose=2
        )
    else:
        x, y = get_model_data(data)
        show_model_angles_distribution(y)

        history_object = model.fit(
            x,
            y,
            validation_split=0.2,
            shuffle=True,
            epochs=5,
            batch_size=32,
            verbose=2
        )

save_model(model)
show_model_performance(history_object)
