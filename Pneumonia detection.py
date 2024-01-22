#!/usr/bin/env python
# coding: utf-8

# ### Data Understanding

# The dataset is structured into three main folders: train, test, and val. Each folder contain subfolders for two distinct image categories—Pneumonia and Normal. 
# Within these folders, a total of 5,856 chest X-ray images in JPEG format are present. The images, captured using the anterior-posterior technique, originate from retrospective cohorts of pediatric patients aged one to five years at the Guangzhou Women and Children’s Medical Center in Guangzhou.
# 
# The inclusion of these chest X-ray images is the dataset was part of routine clinical care for the pediatric patients. To ensure the dataset's quality, an initial screening process was conducted to eliminate low-quality or unreadable scans. Subsequently, two expert physicians meticulously graded the diagnoses associated with the images. Only after this rigorous evaluation were the images deemed suitable for training the AI system.
# 
# In order to mitigate potential grading errors, a further layer of scrutiny was applied to the evaluation set. This involved an examination by a third expert, adding an extra level of assurance to the accuracy of the diagnoses. This comprehensive approach to quality control and grading ensures a robust foundation for the analysis of chest X-ray images and enhances the reliability of the AI system trained on this dataset.

# #### Importing libraries

# In[8]:


# Import libraries
import os
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
import opendatasets as od

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import random
from pathlib import Path #to be able to use functions using path


# Data science tools
import pandas as pd # data processing
import numpy as np # linear algebra

# Tensorflow for GPU
import tensorflow as tf
from tensorflow.compat.v1 import Session, ConfigProto, set_random_seed
from tensorflow.python.client import device_lib

# Keras library for Modeling
import keras
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.constraints import max_norm
#import tensorflow.contrib.keras as keras
from keras import backend as K

# OpenCV
import cv2

# Resize images
from skimage.io import imread
from skimage.transform import resize

# Scikit-learn library
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Visualizations
from PIL import Image
import imgaug as aug
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import matplotlib.image as mimg # images
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px


# #### Loading Data
# First we load the dataset of chest X-ray images with labeled annotations indicating whether each image contains pneumonia or is normal. Then we resize all images to a consistent size to ensure uniformity.

# In[23]:


# Loading and resizing images

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_arr is not None:
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            else:
                print(f"Error loading image: {img_path}")
    return np.array(data, dtype=object)


# loading data
train_data_folder = 'E:/Chest Xray/chest_xray/train'
test_data_folder = 'E:/Chest Xray/chest_xray/test'
val_data_folder = 'E:/Chest Xray/chest_xray/val'


# #### To find the number of images in the training, testing, and validation sets

# In[26]:


# Function to count the number of image files in a folder
def count_images_in_folder(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

# Count the number of images in each dataset
train_images = count_images_in_folder(os.path.join(train_data_folder, 'PNEUMONIA')) + \
                count_images_in_folder(os.path.join(train_data_folder, 'NORMAL'))

test_images = count_images_in_folder(os.path.join(test_data_folder, 'PNEUMONIA')) + \
               count_images_in_folder(os.path.join(test_data_folder, 'NORMAL'))

val_images = count_images_in_folder(os.path.join(val_data_folder, 'PNEUMONIA')) + \
              count_images_in_folder(os.path.join(val_data_folder, 'NORMAL'))

# Display the counts
print(f"Number of images in the training set: {train_images}")
print(f"Number of images in the testing set: {test_images}")
print(f"Number of images in the validation set: {val_images}")


# #### To check and print the count of files in different subdirectories corresponding to the specified labels for the training, validation, and test sets.

# In[27]:


def count_file(dir=None, labels=None):
  for label in labels:
    num_data= len(os.listdir(os.path.join(dir, label)))
    print(f'Count of {label} : {num_data}')
    
labels= ['PNEUMONIA','NORMAL']
print('Train Set: \n' + '='*50)
count_file(train_data_folder,labels)

print('\nValidation Set: \n' + '='*50)
count_file(val_data_folder,labels)

print('\nTest Set: \n' + '='*50)
count_file(test_data_folder,labels)


# In[ ]:




