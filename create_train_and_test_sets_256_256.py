'''

Создание набора патчей для тренировки нейронной сети UNET
Подробности в документе "Сегментация на основе UNET.docx"

'''

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from patchify import patchify, unpatchify
from skimage.io import imshow
from numpy.random import default_rng

PATH = 'D:\Datasets\ElectronMicroscopyDataset'

os.chdir(PATH)

def sanity_check(image_set = 'train',number_of_images = 4):
    '''
    Выводит несколько изображений и соответствующие им маски

    Parameters
    ----------
    image_set : TYPE, optional
        DESCRIPTION. The default is 'train'.

    Returns
    -------
    None.

    '''
    
    rng = default_rng()
    patches = os.path.join(PATH,'{}_256_256'.format(image_set))
    masks = os.path.join(PATH,'{}_groundtruth_256_256'.format(image_set))
    try:
        assert os.path.exists(patches) and os.path.exists(masks)
        assert len(os.listdir(patches)) == len(os.listdir(masks))
        
        idx = 1
        
        rand_ints = rng.integers(low = 0,high = len(os.listdir(patches)),size = number_of_images)
        plt.figure()
        plt.suptitle(image_set)
        for rand_int in rand_ints:
            rand_patch = Image.open(os.path.join(patches,'image_{}.tif'.format(rand_int)))
            rand_mask = Image.open(os.path.join(masks,'mask_{}.tif'.format(rand_int)))
            plt.subplot(number_of_images,2,idx)
            plt.imshow(rand_patch,cmap = 'gray')
            plt.title('image_{}'.format(rand_int))
            plt.xticks([])
            plt.yticks([])
            idx += 1
            plt.subplot(number_of_images,2,idx)
            plt.imshow(rand_mask,cmap = 'gray')
            plt.title('mask_{}'.format(rand_int))
            plt.xticks([])
            plt.yticks([])
            idx += 1
        plt.show()
    except Exception as e:
        print("Exception is: {}".format(e.__class__))
        
        
def crop_images_with_pil(w = 256,h = 256,path = '',image_set = 'train',image_type = ''):
    '''
    
    Процедура по фрагментации изображений при помощи PIL
    
    Parameters
    ----------
    w : int
        Width of the crop. The default is 256.
    h : int
        Height of the crop. The default is 256.
    path: str
        Original image path
    image_type: str
        Can be 'train' or 'test'
    '''
    try:
        assert image_set == 'train' or image_set == 'test'
        assert image_type == '' or image_type == 'groundtruth'
        
        if image_type == 'groundtruth':
            split_path = os.path.join(path,'{}_{}_split'.format(image_set,image_type))
        else:
            split_path = os.path.join(path,'{}_{}split'.format(image_set,image_type))
        
        idx = 0
        for _,image in tqdm(enumerate(os.listdir(split_path)),total = len(os.listdir(split_path))):
            img = Image.open(os.path.join(split_path,image))
            for i in range(np.int(img.width/256)):
                for j in range(np.int(img.height/256)):
                    img_crop = img.crop((i * 256,j * 256,i * 256 + 256,j * 256 + 256)) # create 256x256 crop of image  (left,upper,right,lower)
                    
                    if image_type == 'groundtruth':
                        img_crop_path = os.path.join(PATH,'{}_{}_256_256'.format(image_set,image_type))
                    else:
                        img_crop_path = os.path.join(PATH,'{}_{}256_256'.format(image_set,image_type))
                    
                    if not os.path.exists(img_crop_path):
                        print('    creating directory    ' + img_crop_path)
                        os.makedirs(img_crop_path)
                    
                    if image_type == 'groundtruth':
                        img_crop.save(os.path.join(img_crop_path,'mask_{}.tif'.format(idx)))
                    if image_type == '':
                        img_crop.save(os.path.join(img_crop_path,'image_{}.tif'.format(idx)))
                    
                    idx += 1
    except Exception as e:
        print("Exception is: {}".format(e.__class__))

def crop_images_with_patchify(w = 256,h = 256, path = '', image_set = 'train',image_type = ''):
    '''
    
    Фрагментация изображений при помощи patchify
    
    Parameters
    ----------
    w : int
        Width of the crop. The default is 256.
    h : int
        Height of the crop. The default is 256.
    path: str
        Original image path
    image_type: str
        Can be 'train' or 'test'

    '''
    try:
        assert image_set == 'train' or image_set == 'test'
        assert image_type == '' or image_type == 'groundtruth'
        
        if image_type == 'groundtruth':
            split_path = os.path.join(path,'{}_{}_split'.format(image_set,image_type))
        else:
            split_path = os.path.join(path,'{}_{}split'.format(image_set,image_type))
        idx = 0
        
        for _,image in tqdm(enumerate(os.listdir(split_path)),total = len(os.listdir(split_path))):
            image = cv2.imread(os.path.join(split_path,image),cv2.IMREAD_GRAYSCALE)
            patches = patchify(image,(256,256),step = 256)
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    img_crop = Image.fromarray(patches[i,j,:,:])
                    
                    if image_type == 'groundtruth':
                        img_crop_path = os.path.join(PATH,'{}_{}_256_256'.format(image_set,image_type))
                    else:
                        img_crop_path = os.path.join(PATH,'{}_{}256_256'.format(image_set,image_type))
                    
                    if not os.path.exists(img_crop_path):
                        print('    creating directory    ' + img_crop_path)
                        os.makedirs(img_crop_path)
                    
                    if image_type == 'groundtruth':
                        img_crop.save(os.path.join(img_crop_path,'mask_{}.tif'.format(idx)))
                    if image_type == '':
                        img_crop.save(os.path.join(img_crop_path,'image_{}.tif'.format(idx)))
                    
                    idx += 1
    except Exception as e:
        print("Exception is: {}".format(e.__class__))
    

                
if __name__ == '__main__':
    # TODO: переориентировать процедуры по фрагментации на изображения разного размера
    # parser = argparse.ArgumentParser(description = 'Cropping of image script')
    # parser.add_argument('--width', default = 256)
    # parser.add_argument('--height', default = 256)
    # parser.add_argument('--save_dir',default = os.getcwd())
    # args = parser.parse_args()
    print("Create train set...")
    crop_images_with_patchify(image_set = 'train',image_type = '')
    crop_images_with_patchify(image_set = 'train',image_type = 'groundtruth')
    print("Create test set...")
    crop_images_with_patchify(image_set = 'test',image_type = '')
    crop_images_with_patchify(image_set = 'test',image_type = 'groundtruth')
    print("Sanity check...")
    sanity_check(image_set = 'train')
    sanity_check(image_set = 'test')
