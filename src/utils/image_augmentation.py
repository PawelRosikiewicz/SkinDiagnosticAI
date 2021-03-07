# ********************************************************************************** #
#                                                                                    #
#   Project: FastClassAI workbecnch                                                  #                                                  
#                                                                                    #                      
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz_gmail.com                                                  #
#                                                                                    #
#   This notebook is a part of Skin AanaliticAI development kit, created             #
#   for evaluation of public datasets used for skin cancer detection with            #
#   large number of AI models and data preparation pipelines.                        #
#                                                                                    #     
#   License: MIT                                                                     #
#   Copyright (C) 2021.01.30 Pawel Rosikiewicz                                       #
#   https://opensource.org/licenses/MIT                                              # 
#                                                                                    #
# ********************************************************************************** #



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os # allow changing, and navigating files and folders, 
import sys
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import random # functions that use and generate random numbers

import numpy as np # support for multi-dimensional arrays and matrices
import pandas as pd # library for data manipulation and analysis
import seaborn as sns # advance plots, for statistics, 
import matplotlib.pyplot as plt # for making plots, 
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import tensorflow as tf
import tensorflow_hub as hub
import scipy.stats as stats  # library for statistics and technical programming, 
import tensorflow.keras as keras  

from PIL import Image, ImageDraw
from IPython.display import display
from tensorflow.keras import backend as K # used for housekeeping of tf models,

import matplotlib.patches as mpatches
from tensorflow.keras.preprocessing.image import ImageDataGenerator




# Function ................................................................................

def create_augmented_images(*, external_generator, augm_img_nr=10, paramsforgenerator=""):
    """ 
        Function that takes pictures in a batch, provided with keras generators
        and uses another generator.
        Secondarly, this function can be used to create dataframe with data on images in image batch
        if, augm_img_nr is set 0, 
        
        external_generator     : iterator, based on keras image generator
                                 the function was designed to work with all images in a given dataset
                                 provided as one batch,
        
        augm_img_nr            : the number of augment images that will be created 
                                 for each image, if augm_img_nr=0, no augmented images will be created, 
                                 but both array, and dataframe will be returned, 
        
        paramsforgenerator     : dictionary, with parameters for image generator,
                                 used for image augmentation, 
                                 
        Returns                : numpy array with img batch, [?, pixel_size, pixel_size, 3]
                                 pandas dataframe, with rows corresponding to each image in the batch, 
                                 and following columns: 
                                 class = foldername in data directory, imagename= original image name, 
                                 imgtype={'raw', 'aug'}, imgidnumber=0, foir raw, >=1 for augmented images
    """

    # extract one batch with all images in a given dataset
    img_batch, batch_labels = next(external_generator)

    #.. create df, with class, image and image type names
    """ I will use this df, to create, new file with subdirectories, 
        and save raw and augmented images with proper names
    """
    img_filenames = pd.Series(external_generator.filenames).str.split(pat="/", expand=True)
    img_filenames = pd.concat([img_filenames, pd.Series(["raw"]*img_filenames.shape[0]), pd.Series([0]*img_filenames.shape[0])], axis=1)
    img_filenames.columns = ["classname", "imgname", "imgtype", "imgidnumber" ]

    # in case, I just wish to use that function to get everythign in the same format, but not to generate augmented images
    if augm_img_nr==0: 
      pass
    
    if augm_img_nr>0:
    
      # Create generator for image augmentation
      datagen = ImageDataGenerator(**paramsforgenerator)
      datagen.fit(img_batch)

      #.. prepare iterator, that will return all figures in a batch, one by one, 
      # augm_datagen.fit(img_batch)
      datagen_iter =  datagen.flow(img_batch, batch_size=1, shuffle=False) 


      # Create n augmented figures for each image in gthe batch, 
      aug_img_filenames = list()
      for i in range(augm_img_nr):
          for j in range(img_batch.shape[0]):
              # create augmented figure, and add to new batch
              one_img = datagen_iter.next()
              if i+j==0: 
                  batch_img_augm = one_img
              else: 
                  batch_img_augm = np.r_[batch_img_augm, one_img]

              # save name and id for that image
              aug_img_filenames.append({
                  "classname" : img_filenames.iloc[j,0],
                  "imgname": img_filenames.iloc[j,1], 
                  "imgtype": "aug",
                  "imgidnumber": i+1})            
    
      # create new batch and df with labels and filenames to return,
      img_filenames = pd.concat([img_filenames,pd.DataFrame(aug_img_filenames)], axis=0, sort=False).reset_index(drop=True)
      img_batch     = np.r_[img_batch, batch_img_augm]
        
    #print(img_filenames.shape, img_batch.shape)
    return img_batch, img_filenames
  
  
  
  
  
# Function ................................................................................
  
def save_augmented_images(*,
    datasetname, img_batch, batch_info, savedir, verbose=False):

    """
        1) creates save directory, with subdirectories for saving classified images
        2) saves images as png, that were stored in img_batch
        
        datasetname    : str, eg {"test", "train"}
        img_batch.     : numpy array [?, pixel_nr, pixel_nr, 3], contains rgb pictures 
                         on scale [0-255]
        batch_info     : data frame with info on each image in img_batch
                         created with create_augmented_images()
        savedir        : full path to directory, where all classes should be stored, 
        verbose        : default = False,    
    """

    # check if savedir exist, if not create it
    try: os.chdir(savedir)
    except: os.mkdir(savedir)

    # create directories with provided datasetname
    os.chdir(savedir)
    try: os.mkdir(datasetname)
    except: pass

    # create directories for each class
    os.chdir(os.path.join(savedir, datasetname))
    for dirname in list(batch_info.classname.unique()):
        try: os.mkdir(dirname)
        except: pass 

    # save each images in img_batch with proper name in corresponing class/directory
    for i in range(img_batch.shape[0]):
        img_info = batch_info.iloc[i,:]

        # img name
        if img_info.imgtype=="raw":
            img_name = f"{img_info.imgtype}_{img_info.imgname}"
        if img_info.imgtype!="raw":
            img_name = f"{img_info.imgtype}{img_info.imgidnumber}_{img_info.imgname}"

        # saving, 
        try:
            mpl.image.imsave(os.path.join(savedir, datasetname, img_info.classname, img_name), 
                                 np.array(img_batch[i], dtype=int)
                                ) # [0-255] must be int, 
        except: 
            pass

    # info,
    if verbose==True:
        print(f"{img_batch.shape[0]} images were saved")
        print(f"in {savedir}")
        print(f"in following files for each classe: {list(batch_info.classname.unique())}")

