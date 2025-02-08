import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

data_dir = 'data/Imagens'
img_height, img_width = 150, 150
batch_size = 32
num_classes = len(os.listdir(data_dir))
epochs = 25

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

