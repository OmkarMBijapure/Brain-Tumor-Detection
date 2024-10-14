import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical


#reading dataset
image_directory = 'datasets/'
no_tumorimages = os.listdir(image_directory+ 'no/')
yes_tumorimages = os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]

INPUT_SIZE = 64

#print(no_tumorimages)
#path = 'no0.jpg'
#print(path.split('.')[1])

for i, image_name in enumerate(no_tumorimages):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumorimages):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

print(len(dataset))
print(len(label))

dataset =np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)

#Reshape = (n, image_width, image_height, n_channel)

print(x_train.shape)#(80% of data = 2400 images)
print(y_train.shape)#(80% of data = 2400 images)

print(x_test.shape)#(20% of data = 600 images)
print(y_test.shape)#(20% of data = 600 images)

x_train= normalize(x_train, axis =1)
x_test= normalize(x_test, axis =1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
#Model Building

model=Sequential()
#64,64,3
model.add(Conv2D(32,(3,3),input_shape=(INPUT_SIZE, INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) # 1 bcs our output is yes or no , and we are using binary classification problem
model.add(Activation('softmax'))


#Binary CrossEntropy =1 , sigmoid
#Categorical Cross Entropy = 2 , softmax

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, 
          batch_size= 20,
          verbose =1 ,epochs=10,
          validation_data=(x_test,y_test),
          shuffle = False)

model.save('BrainTumor10EpochsCategorical.h5')


















