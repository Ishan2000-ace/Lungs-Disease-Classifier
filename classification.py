# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:52:09 2020

@author: Ishan Nilotpal
"""

from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob

Image_Size = [224,224]

train_path = 'C:/Users/hp/Documents/Machine Learning Projects/Pneumonia classification/chest_xray/train'
test_path = 'C:/Users/hp/Documents/Machine Learning Projects/Pneumonia classification/chest_xray/test'

vgg = VGG16(input_shape=Image_Size+[3],weights='imagenet',include_top=False)

for layer in vgg.layers:
    layer.trainable = False
    
folders = glob('C:/Users/hp/Documents/Machine Learning Projects/Pneumonia classification/chest_xray/train/*')

x = Flatten()(vgg.output)

prediction = Dense(len(folders),activation='softmax')(x)

model = Model(inputs=vgg.input,outputs=prediction)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_path,target_size=(224,224),batch_size=32,class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_path,target_size=(224,224),batch_size=32,class_mode='categorical')

r = model.fit_generator(training_set,validation_data=test_set,epochs=5,steps_per_epoch=len(training_set),validation_steps=len(test_set))

model.save('Lungs.h5')
print('Saved Model to disk')

