# -*- coding: utf-8 -*-
"""

Тренировка нейронной сети UNET для сегментации митохондрий
Подробности в документе "Сегментация на основе UNET.docx"

"""



import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from skimage.io import imshow
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import mitochondria_nn_model as unet_model

# В случае использования Google Colab нужно добавить в path директорию, в которой находится
# скрипт с моделью

sys.path.append(os.path.join(os.getcwd(),'drive/MyDrive/NN_models'))

# В случае использования Google Colab изображения первоначально хранятся
# в формате *.zip (для удобства копирования на Google Disk)

import zipfile

train_zip_path = os.path.join(os.getcwd(),'drive/MyDrive/Datasets/training_256_256.zip')
test_zip_path = os.path.join(os.getcwd(),'drive/MyDrive/Datasets/testing_256_256.zip')
train_groundtruth_zip_path = os.path.join(os.getcwd(),'drive/MyDrive/Datasets/training_groundtruth_256_256.zip')
test_groundtruth_zip_path = os.path.join(os.getcwd(),'drive/MyDrive/Datasets/testing_groundtruth_256_256.zip')

zip_file = zipfile.ZipFile(train_zip_path,'r')
zip_file.extractall(os.path.join(os.getcwd(),'drive/MyDrive/Datasets/'))
zip_file.close()

zip_file = zipfile.ZipFile(test_zip_path,'r')
zip_file.extractall(os.path.join(os.getcwd(),'drive/MyDrive/Datasets/'))
zip_file.close()

zip_file = zipfile.ZipFile(train_groundtruth_zip_path,'r')
zip_file.extractall(os.path.join(os.getcwd(),'drive/MyDrive/Datasets/'))
zip_file.close()

zip_file = zipfile.ZipFile(test_groundtruth_zip_path,'r')
zip_file.extractall(os.path.join(os.getcwd(),'drive/MyDrive/Datasets/'))
zip_file.close()


# Формирование тензоров для тренировки и тестирования

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = os.path.join(os.getcwd(),"drive/MyDrive/Datasets/training_256_256")
TRAIN_GROUNDTRUTH_PATH = os.path.join(os.getcwd(),"drive/MyDrive/Datasets/training_groundtruth_256_256")
TEST_PATH = os.path.join(os.getcwd(),"drive/MyDrive/Datasets/testing_256_256")
TEST_GROUNDTRUTH_PATH = os.path.join(os.getcwd(),"drive/MyDrive/Datasets/testing_groundtruth_256_256")

train_files = next(os.walk(TRAIN_PATH))[2] # get file names
train_groundtruth_files = next(os.walk(TRAIN_GROUNDTRUTH_PATH))[2] # get file names
test_files = next(os.walk(TEST_PATH))[2] # get file names
test_groundtruth_files = next(os.walk(TEST_GROUNDTRUTH_PATH))[2] # get file names

X_train = np.zeros((len(train_files),IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS),dtype = np.uint8)
y_train = np.zeros((len(train_groundtruth_files),IMG_WIDTH,IMG_HEIGHT,1),dtype = np.bool) # groundtruth

X_test = np.zeros((len(train_files),IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS),dtype = np.uint8)
y_test = np.zeros((len(train_groundtruth_files),IMG_WIDTH,IMG_HEIGHT,1),dtype = np.bool) # groundtruth

for idx,train_name in tqdm(enumerate(train_files),total = len(train_files)):
  X_train[idx] = cv2.imread(os.path.join(TRAIN_PATH,train_name),cv2.IMREAD_COLOR)

for idx,train_groundtruth_name in tqdm(enumerate(train_groundtruth_files),total = len(train_groundtruth_files)):
  image = cv2.imread(os.path.join(TRAIN_GROUNDTRUTH_PATH,train_groundtruth_name),cv2.IMREAD_GRAYSCALE)
  bool_image = image > 0
  y_train[idx] = bool_image[:,:,np.newaxis]

for idx,test_name in tqdm(enumerate(test_files),total = len(test_files)):
  X_test[idx] = cv2.imread(os.path.join(TEST_PATH,test_name),cv2.IMREAD_COLOR)

for idx,test_groundtruth_name in tqdm(enumerate(test_groundtruth_files),total = len(test_groundtruth_files)):
  image = cv2.imread(os.path.join(TEST_GROUNDTRUTH_PATH,test_groundtruth_name),cv2.IMREAD_GRAYSCALE)
  bool_image = image > 0
  y_test[idx] = bool_image[:,:,np.newaxis]

# Для удобства можно сохранить считанные изображения, а в дальнейшем
# работать уже сразу с сформированными массивами

import pickle

# Сохранение изображений

pickle_out = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/X_train.pckl"),'wb')
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/y_train.pckl"),'wb')
pickle.dump(y_train,pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/X_test.pckl"),'wb')
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/y_test.pckl"),'wb')
pickle.dump(y_test,pickle_out)
pickle_out.close()

# Загрузка сохраненных изображений

pickle_in = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/X_train.pckl"),'rb')
X_train_pckl = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/y_train.pckl"),'rb')
y_train_pckl = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/X_test.pckl"),'rb')
X_test_pckl = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join(os.getcwd(),"drive/MyDrive/ElectronMicroscopyDataset/y_test.pckl"),'rb')
y_test_pckl = pickle.load(pickle_in)
pickle_in.close()

# Количество тренировочных изображений должно быть больше, чем тестовых
# поэтому создаем новые массивы. В данном случае изображения были сначала
# сохранены с помощью pickle.
# Вместо числа 1500 можно полностью соединить массивы, а затем использовать scikit-learn метотд train_test_split,
# в котором указать требуемую границу для разделения тренировочных и тестовых данных

X_train = np.concatenate((X_train_pckl,X_test_pckl[:1500,...]))
X_test = X_train_pckl[1500:,...]
y_train = np.concatenate((y_train_pckl,y_test_pckl[:1500,...]))
y_test = y_train_pckl[1500:,...]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# проверка изображений

imshow(X_train[52])
plt.figure()
imshow(y_train[52].squeeze())

# Создание модели U-Net

# Create the unet model
model = unet_model.mitochondria_nn_model(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)

# Создание обратных вызовов.
# Позволяют остановить тренировку, когда потери на валидации перестают изменяться,
# сохранить лучшую модель в файл mitochondria_model.h5, а также в дальнейшем посмотреть графики точности и потери
# с помощью TensorBoard


callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 2),
    tf.keras.callbacks.TensorBoard(log_dir = 'logs'),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(),"drive/MyDrive/NN_models/mitochondria_model.h5"), verbose = True, save_best_only = True)
    ]

# Тренировка модели. На CPU происходит очень долго. Если нет GPU, то следует использовать Google Colab

results = model.fit((X_train), y_train, validation_split = 0.1, batch_size = 16, epochs = 50, callbacks = callbacks)

# Просмотр графиков точности и потери при помощи TensorBoard (при использовании Google Colab)
# %load_ext tensorboard
# %tensorboard --logdir logs

# Выполнение предсказаний. На выходе вероятность значений.
y_pred = model.predict(X_test)

# По факту здесь производится выделение первой картинки
# Из массива. Последний индекс 0 можно не использовать, если выполнить y_pred.squeeze()
# В ситуации, когда X_test состоит только из одного элемента, получение предсказанного изображения
# может иметь следующий вид:
# predicted_tensor = model.predict(test_image)
# squeezed_tensor = predicted_tensor.squeeze()
# pred_image = (squeezed_tensor > 0.5).astype(np.uint8)

pred_img = (y_pred[1,:,:,0] > 0.5).astype(np.uint8)
imshow(pred_img,cmap = 'gray')
plt.figure()
imshow(y_test[1].squeeze())

# Нахождение IOU
y_pred_thresh = y_pred > 0.5
intersection = np.logical_and(y_test,y_pred_thresh)
union = np.logical_or(y_test,y_pred_thresh)
iou_score = np.sum(intersection) / np.sum(union)
print("IOU score is: {}".format(iou_score))





