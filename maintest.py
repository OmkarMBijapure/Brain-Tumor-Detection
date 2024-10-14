import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('D:\\personal project\\brain tumor dtetction\\BrainTumor Classification DL\\BrainTumor10Epochs.h5')

image = cv2.imread('D:\\personal project\\brain tumor dtetction\\pred\\pred0.jpg')

img = Image.fromarray(image)

img = img.resize((64,64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

predictions = model.predict(input_img)
predicted_classes = (predictions > 0.5).astype('int32')
print(predicted_classes)




