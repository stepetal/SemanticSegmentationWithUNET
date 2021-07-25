# -*- coding: utf-8 -*-
"""

Тестирование предтренерованнной U-Net

"""

import tensorflow as tf
import numpy as np
from skimage.io import imshow
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from mitochondria_nn_model import mitochondria_nn_model
import os
from numpy.random import default_rng

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

rng = default_rng()


model = mitochondria_nn_model(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)
model_path = os.path.join(os.getcwd(),'pretrained_models')
model.load_weights(os.path.join(model_path,'mitochondria_model_89.h5'))

test_image_path = os.path.join(os.getcwd(),'test_256_256')
test_groundtruth_image_path = os.path.join(os.getcwd(),'test_groundtruth_256_256')

img_number = rng.integers(low = 1500,high = len(os.listdir(test_image_path)))

test_image = cv2.imread(os.path.join(test_image_path,'image_{}.tif'.format(img_number)))
test_groundtruth_image = cv2.imread(os.path.join(test_groundtruth_image_path,'mask_{}.tif'.format(img_number)))


test_image = test_image[np.newaxis,:,:,:]

predicted_tensor = model.predict(test_image)
squeezed_tensor = predicted_tensor.squeeze()
predicted_image = (squeezed_tensor > 0.55).astype(np.uint8)


plt.subplot(1,3,1)
plt.imshow(test_image.squeeze(),cmap = 'gray')
plt.title('Image: image_{}'.format(img_number))
plt.subplot(1,3,2)
plt.title('Groundtruth image: image_{}'.format(img_number))
plt.imshow(test_groundtruth_image.squeeze(),cmap = 'gray')
plt.subplot(1,3,3)
plt.title('Predicted image'.format(img_number))
plt.imshow(predicted_image,cmap = 'gray')

