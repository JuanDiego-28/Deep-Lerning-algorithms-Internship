#!/bin/bash 

"""

author:
date : 25-07-2024 



"""
# imort libs

import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array

from model1 import unet     # repalce model1 with the name you stored the unet_model file 


# def time stamps 
timestamp= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
timestamp  

#---------------------------------------
# define what are you going to train 
#---------------------------------------
FORMAT= 'png'                             # replace with format your are using 
STAIN='norm'
GRID_SIZE= 512
SIZE=GRID_SIZE
BATCH_SIZE=10                             # adjusted for laptop. Original was 16 
MAJOR_KERNEL_SIZE=3
LEARNING_RATE=0.001                       # ajusted for laptopn. Original was 0.0001


# replace with your PATH
TILES_TRAIN_PATH='/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/lympho_tiles/train/'   
TILES_TEST_PATH= '/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/lympho_tilestest_tiles/'
TILES_VALIDATION_PATH= "/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/lympho_tiles/validation_tiles/"

#---------------
# categories
#---------------
image_categories = ['B_lymphoma' , 'T_lymphoma']
NET_NAME='unet_{}.hdf5'.format(timestamp)




#-------------------
# train set 
#--------------------
train_datagen = ImageDataGenerator(
    rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    TILES_TRAIN_PATH,
    target_size=(SIZE, SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    classes= image_categories,
    class_mode='categorical',
    shuffle=True,
    seed=42)                       # the output should be:  found -**amount of images you have** -belonging to- **amount of classes you define** 


#--------------------
# validation set
# ------------------- 
validation_datagen = ImageDataGenerator(
    rescale=1/255)

validation_generator = validation_datagen.flow_from_directory(
    TILES_VALIDATION_PATH,
    target_size=(SIZE, SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    classes= image_categories,
    class_mode='categorical', # None?
    shuffle=True,
    seed=42)                        # the output should be:  found -**amount of images you have** -belonging to- **amount of classes you define** 


#-----------------
# UNET model 
#------------------
SIZE= GRID_SIZE
major_kernel_size=MAJOR_KERNEL_SIZE
learning_rate= LEARNING_RATE

model = unet(SIZE, major_kernel_size, image_categories, learning_rate)


#---------------------------
# training the model
#-----------------------------
print("Start Fit")
model.model.fit(train_generator,
                     steps_per_epoch=8800,
                     epochs=5, # 15
                     validation_data=validation_generator,
                     #callbacks=[model_checkpoint],            unoconmet if you want the callbacks 
                     validation_steps=1000)
print("End Fit")

#------------
#Test set 
#------------

test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(
    TILES_TEST_PATH,
    target_size=(SIZE, SIZE),
    color_mode="rgb",
    classes= image_categories,
    class_mode=None,
    shuffle=False,
    batch_size=1,)

filenames = test_generator.filenames
test_samples = len(filenames)
#--------------
#check results
#--------------
test_results = model.predict(test_generator, steps= test_samples, verbose=1)
