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
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import tensorflow_hub as hub

import tensorflow as tf # tf.__version__ 
import tensorflow.keras as keras 
import matplotlib.pyplot as plt # for making plots, 
import scipy.stats as stats  # library for statistics and technical programming, 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw
from IPython.display import display
from tensorflow.keras import backend as K # used for housekeeping of tf models,










# Function, ........................................................

def load_encoded_imgbatch_using_logfile(*, logfile_name, load_datasetnames, verbose=False):
    """
        Function used to load and concatenate encoded images in several batches 
        using logfiles created before, 
        I used that functions to load test and valid data subsets
        # ....
        logfile_name      : str, path to logfile
        load_datasetnames : list, eg: ["test", "train"], these two dastasets will be concastenated in that order        
    """
    # load files & select requested subsets, 
    one_logfile = pd.read_csv(logfile_name)
    one_logfile = one_logfile.set_index("datasetname")
    one_logfile = one_logfile.loc[load_datasetnames]
    one_logfile.reset_index(drop=False, inplace=True)

    # load one or more datasubsets, and join into one array or dataframe, 
    for i in range(one_logfile.shape[0]):
        if i==0:
            encoded_img_batch = np.load(one_logfile.img_batch_features_filename.iloc[i])
            batch_labels = pd.read_csv(one_logfile.img_batch_info_filename.iloc[i])
        if i>0:    
            encoded_img_batch = np.r_[encoded_img_batch, np.load(one_logfile.img_batch_features_filename.iloc[i])]
            batch_labels = pd.concat([batch_labels, pd.read_csv(one_logfile.img_batch_info_filename.iloc[i])], axis=0)
            
        if verbose==True:
            print(f"Loading encoded batch nr {i}, the final batch has now shape of: {encoded_img_batch.shape}")

    # reset indexes in batch_labels
    batch_labels = batch_labels.reset_index(drop=True)
    
    return encoded_img_batch, batch_labels

  
  
  

# Function, ........................................................

def load_raw_img_batch(*, load_datasetnames, path, image_size=(224,224), verbose=False, batch_size=1000, shuffle_images=False):
    
    """
        I added this function to allow plotting examples of images from several datasubests eg train and test together on one plot
        the function will load batch of images, no labels, in and concatenate them in the order given to load_datasetnames,
        it was create to be used to gether with load_encoded_imgbatch_using_logfile, ie the fu ction that loads csv table as opd.dataframe
        with easy reradable labels for images, Caution: to obtain images in the same order, as labels , you must load the same datsets 
        in both functions, and use datsubest in the same order !
    """
    
    # Create Image generator, with rescaling, 
    datagen = ImageDataGenerator(rescale=1/255)

    # collect iterators for each datatype in swissroads, 
    iterators_dct = dict()
    for setname in load_datasetnames:    
        iterators_dct[setname] = datagen.flow_from_directory(
            os.path.join(path, setname),
            batch_size=batch_size, # it will use only max available pictures,  
            target_size=image_size, 
            shuffle=shuffle_images     # here I use generator only to explote the data
        )
        
    # use iterators to load the data and combine in proper order, 
    for i, setname in enumerate(load_datasetnames):
        
        if i == 0: 
            raw_img_batch, _ = next(iterators_dct[setname])
        if i>0:
            another_raw_img_batch, _ = next(iterators_dct[setname])
            raw_img_batch = np.r_[raw_img_batch, another_raw_img_batch]
            
        if verbose==True: print(f"loading batch {i},//{setname}//, final batch has shape of: {raw_img_batch.shape}")
    
    return raw_img_batch


  



# Function, ........................................................  
  
def load_raw_img_batch_and_batch_labels(*, load_datasetnames, path, image_size=(224,224), verbose=False, batch_size=1000, shuffle_images=False):
    
    """
        I added this function to allow plotting examples of images from several datasubests eg train and test together on one plot
        the function will load batch of images, no labels, in and concatenate them in the order given to load_datasetnames,
        it was create to be used to gether with load_encoded_imgbatch_using_logfile, ie the fu ction that loads csv table as opd.dataframe
        with easy reradable labels for images, Caution: to obtain images in the same order, as labels , you must load the same datsets 
        in both functions, and use datsubest in the same order !
        
        Caution
        This function assumes that all batches loaded have the same class compostion, 
        if not you shdoul use different approach, or use batch labels prepared befre dividing 
        images into batches, 
        
        returns:
        . raw_img_batch           : array with images,
        . batch_labels_index      : 2d array with 0/1 class assigment 1==class, row=image, col = class
        . class_indices           : dict, key=Class_name, value=row_nr in batch_labels_index,
        . batch_labels            : list, only with batch labels of loaded images, !
    """
    
    # Create Image generator, with rescaling, 
    datagen = ImageDataGenerator(rescale=1/255)

    # collect iterators for each datatype in swissroads, 
    iterators_dct = dict()
    for setname in load_datasetnames:    
        iterators_dct[setname] = datagen.flow_from_directory(
            os.path.join(path, setname),
            batch_size=batch_size, # it will use only max available pictures,  
            target_size=image_size, 
            shuffle=shuffle_images     # here I use generator only to explote the data
        )
            
    # use iterators to load the data and combine in proper order, 
    for i, setname in enumerate(load_datasetnames):
        
        if i == 0: 
            raw_img_batch, batch_labels_index  = next(iterators_dct[setname]) # only here now batch_labels_index
            class_indices = iterators_dct[setname].class_indices
        if i>0:
            another_raw_img_batch, another_batch_labels_index = next(iterators_dct[setname])
            raw_img_batch      = np.r_[raw_img_batch, another_raw_img_batch]
            batch_labels_index = np.r_[batch_labels_index, another_batch_labels_index]
            
        if verbose==True: 
            print(f"loading batch {i},//{setname}//, final batch has shape of: {raw_img_batch.shape}")
        else:
            pass
            
    # turn array with 0/1 for class into list wiht class labels, 
    bli_df = pd.DataFrame(batch_labels_index)
    for k,v in class_indices.items():
        classcol = bli_df.iloc[:,v]
        if verbose==True: 
            print(k, v,  (classcol==1).sum())
        else:
            pass
        bli_df.loc[classcol==1,v] = k

    # replace all remaning zeros with Nan are remove them to keep only ordered image labels, 
    bli_df[bli_df==0]=np.nan  
    batch_labels = bli_df.unstack().dropna().values.tolist()
            
            
    return raw_img_batch, batch_labels_index, class_indices, batch_labels


  
  
  
  
  
  
  
  
  
  
  
  
# Function, .........................................................................................
  
def load_raw_img_batch_with_custom_datagen(*,     
    path, 
    subset_names,
    n_next_datagen=1,                              
    # --- ImageDataGenerator_kwargs
    ImageDataGenerator_kwargs = None,                        
    # --- for datagen.flow_from_directory()
    datagen__target_size=(256,256),            
    # --- generator stuff, solved with my script, 
    subset_batch_size = 50,  # only max will be used,                                         
    shuffle_batch = False,
    shuffle_all = False,
    # --- function, 
    verbose=False
    ):                                            
                                              
    """
        Loads and combines img_batches with sutom size batches, 
        Caution: datagen = ImageDataGenerator(rescale=1/255)  wotk only for RGB images, !
        
        .........................................................................................
        
        # inputs, 
        . path                : full path to file that contains direcory names with at least one dataset to load, 
                                see keys in generator_dict --- Not used if custom generators are provided! 
        . subset_names        : list with strings, =file name with data grouped in folders with classnames,        
        . n_next_datagen      : int, >0, number of times each iterator is called on the data,         

        # --- ImageDataGenerator_kwargs
        ImageDataGenerator_kwargs: dict, with parameters for keras ImageDataGenerator

        # --- for datagen.flow_from_directory()
        datagen__target_size     : tupple, eg: (256,256) - generator assumes that you use rgb images, with the third dimension
    
        # --- generator stuff,
        subset_batch_size      : batch size loded n times (n_next_datagen) only max will be used,    
                                 Caution, if no augmentation was used, the same images may have been loaded several times, 
                                 with n>1
                                 Caution, if subset_batch_size<number of all images in the subset, 
                                 the parameter shuffle_batch is always ==Truz
                                 
        shuffle_batch          : bool, False, images withint each bat5ch will be mixed, but the batches are not mixed with each other, ¨
                                 and are added as new rows, 
        shuffle_all            : False, images withint each batch will be mixed, but the batches are not mixed with each other, ¨
                                 and are added as new rows,
        # info, 
        verbose                : bool, {True, False}        
        
        
        # NOTES
        . POROBLEM: i need to select raw file names of images taken by each pass, instead i am gettign namess of all files, avaikable, 
                   https://stackoverflow.com/questions/41715025/keras-flowfromdirectory-get-file-names-as-they-are-being-generated

    """
    
    # Part 1. check for iterators, and crearte if nesessarly, .....................................................................
    
    # create data generator, 
    if ImageDataGenerator_kwargs!=None:
        datagen = ImageDataGenerator(rescale=1/255, **ImageDataGenerator_kwargs) 
        imgtype = "augmented"
    else:
        datagen = ImageDataGenerator(rescale=1/255)
        imgtype = "raw"
        
        
    # create data iterator for each subset_name
    dataiter_dict = dict()
    for subset_name in subset_names:
        dataiter_temp  = datagen.flow_from_directory(
                        os.path.join(path, subset_name),
                        batch_size=1, shuffle=False)
        
        # .. find how many iages is there to load all of them, 
        img_nr_in_one_subset = int(len(dataiter_temp.filenames))
        
        # .. create proper iterator, that allowss loading all availble images, - here it will always load all files, 
        dataiter_dict[subset_name]  = datagen.flow_from_directory(
                        os.path.join(path, subset_name), 
                        target_size=datagen__target_size,
                        batch_size=img_nr_in_one_subset, 
                        shuffle=False # done later on by my fucntion        
        )
       
    
    # Part 2. build img_batch and batch labels dataframe, ...............................................................
    
    # use iterators to load the data and combine in proper order, 
    counter = -1 # to start with 0, later on, :)
    for subset_name in subset_names:
        
        for i in range(n_next_datagen):
            counter+=1
            shuffle_one_batch = shuffle_batch # done because some batches may be smaller, and all samples are taken,

            # ..........................................................
            # .. extract images and labels, 
            subset_raw_img_batch, labels = next(dataiter_dict[subset_name])
   
            # .. create df, with class, image and image type names
            img_filenames = pd.Series(dataiter_dict[subset_name].filenames).str.split(pat="/", expand=True)
            subset_batch_labels_df = pd.concat([img_filenames, pd.Series([imgtype]*img_filenames.shape[0]), pd.Series([0]*img_filenames.shape[0])], axis=1)
            subset_batch_labels_df.columns = ["classname", "imgname", "imgtype", "imgidnumber"]
            subset_batch_labels_df["subset_name"] = subset_name
            subset_batch_labels_df["img_idx_in_batch"] = list(range(subset_batch_labels_df.shape[0]))
            subset_batch_labels_df["batch_number"] = i

            
            # ..........................................................
            # my solution to shuffling images having img names            
            
            # .. check for subset_batch_size
            if subset_batch_size!=None:
                if subset_batch_labels_df.shape[0]>=subset_batch_size:
                    batch_size = subset_batch_size
                    # here modify shuffle option, 
                    shuffle_one_batch = True   # because we will take randome sample sumbet from the batch, 
                else:
                    batch_size = subset_batch_labels_df.shape[0] # this way I will not use to many subsamples, 
            else:
                batch_size = subset_batch_labels_df.shape[0]
            
            # .. shuffle and if needed reduce file size, 
            if shuffle_one_batch==True:
                batch_idx = np.arange(subset_batch_labels_df.shape[0])
                new_array_order = np.random.choice(batch_idx, batch_size, replace=False)
                # ...
                subset_raw_img_batch   = subset_raw_img_batch[new_array_order]
                subset_batch_labels_df = subset_batch_labels_df.iloc[new_array_order,:]
            else:
                pass # nothiung is happening, ie, there is no shuffling, and no subset is take, 
                     # you shodul see data from all iages in the sebset loaded, 
                     # caution, this is considered individually, so in case of different subset sizes, 
                     # one may be shuffled and the other not, 
                    
                    
            # ..........................................................
            # finally create or add final dataset, 
            if counter==0:
                raw_img_batch = subset_raw_img_batch 
                batch_labels_df = subset_batch_labels_df
            else:
                raw_img_batch = np.r_[raw_img_batch, subset_raw_img_batch]
                batch_labels_df = pd.concat([batch_labels_df, subset_batch_labels_df], axis=0)
                batch_labels_df.reset_index(drop=True, inplace=True)
        
        
            # ..........................................................
            # and some info to track the progress
            if verbose==True:
                print(f"ADDING batch: {subset_name} * {i+1}, batch_arr: {subset_raw_img_batch.shape}, labels: {subset_batch_labels_df.shape}")
                print(f"------------->>>: final batch_arr: {raw_img_batch.shape}, labels: {batch_labels_df.shape}")
            else:
                pass
    
    # Part 3. Final operations, ...............................................................
    
    if shuffle_all==True:
        all_idx = np.arange(batch_labels_df.shape[0])
        new_order = np.random.choice(all_idx, batch_labels_df.shape[0], replace=False)
        # ...
        raw_img_batch   = raw_img_batch[new_order]
        batch_labels_df = batch_labels_df.iloc[new_order,:]
        
        if verbose==True:
            print("images from all batches and subsets were shiffled, you may reorganize them again using data in data_labels")
        
    else:
        pass 
    
    return raw_img_batch, batch_labels_df           
            
    
    
    
    
    
# Function, .................................................................................................
def cnn_data_loader(
    PATH_raw, 
    train_subset_names,
    random_state_nr = 0, 
    valid_subset_names = None, # if None, it will be created from train data, 
    test_subset_names = None,  # if None, it will not be created at all, 
    model_unit_test = False,   # if True, (train, test and valid datasets are created using Trains datasets without augmentation, in one batch)
    augment_valid_data = False,
    params=None, 
    img_size = None,           # can be used only if params are not given, otherwise it shodul be there, 
    verbose=False
):
    
    
    """
        Prepares train test and valid datasets for model fitting with neural networks, 
        additionally it may prepare a dataset for model_unit test
        
        .........................................................................................
        
        # inputs, 
        . PATH_raw             : full path to file that contains direcory names with at least one dataset to load, 
                                 see keys in generator_dict --- Not used if custom generators are provided! 
        . train_subset_names   : list with strings, =file name with data grouped in folders with classnames,        
        . valid_subset_names   : if None, it will be created from train data, 
        . test_subset_names    : if None, it will not be created at all,       
        . ...
        . params               : dict, 
                                "img_size": (256, 256),
                                "train_test_split__train_size": 0.7, # used only if valid_subset_names==None, 
                                "img_generator":{
                                    "n_next_datagen": int, 
                                    "shuffle_all": bool, 
                                    "ImageDataGenerator_kwargs": dict()
                                for more see help(load_raw_img_batch_with_custom_datagen)
        . augment_valid_data  : bool, if True, valid data loaded from external datasets willl be treated 
                                in the same way as train data, if False, they are loaded as raw images, and only scalled /255
        . model_unit_test     : bool, if True, (train, test and valid datasets are created using 
                               Train dataset without augmentation, in one batch)
        . img_size            : tuple, with two integers eg: (256, 256)
                                if !=None, overrides img_size in params, 
        . verbose             : bool,                      

        # ...................................................................................................
        # example use:

        ImageDataGenerator_kwargs = dict(
            height_shift_range=0.2,
            width_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            rotation_range=2,
            brightness_range=[0.2,1.8], 
            zoom_range=[0.75,1.25], # ±25%
            channel_shift_range=.5
        ) # or None, just for data scaling /255

        data_loader__params = {
            "img_size": (256, 256),
            "train_test_split__train_size": 0.7, # used only if valid_subset_names==None, 
            "img_generator":{
                "n_next_datagen": 2,
                "shuffle_all": True, 
                "ImageDataGenerator_kwargs": ImageDataGenerator_kwargs, # place here a dictionary with parameters to set, 
            }
        }
        
        X_dct, y_dct, batch_labels_dct, idx_y_dct, class_encoding,  class_decoding = cnn_data_loader(
            PATH_raw=PATH_raw,             # fiull path to directory with images prepared for generator, ie one folder == images from one class
                                           #   folder name == class name
            train_subset_names=["train"],  # (I called the main folders train, valid and test, each contians 
                                           #.   the same set of subdirectories with class names and images of that class in each of them)
            valid_subset_names =["valid"], # if None, it will be created from train data, 
            test_subset_names = ["test"],  # if None, it will not be created at all, 
            model_unit_test = False,       # if True, (train, test and valid datasets are created using Trains datasets without augmentation, in one batch)
            augment_valid_data  = True,    # if True, valid data are processed in the same way as train data, 
            img_size = None,               # can be used only if params are not given, otherwise it shodul be there, 
            params=data_loader__params, 
            verbose=False
        )

    """
    
    
    # set up some missing parameters to run the function, 
    
    if params==None: 
        params = dict()
        params["img_size"] = (256, 256)
        params["img_generator"] = None # ie, no augmentation will be done, all imagess will be loaded just once, 
        params["train_test_split__train_size"] = 0.7 # used only if valid_subset_names==None, 
    else:
        pass
    
    if img_size!=None:
        params["img_size"] = img_size

    
    t = True
    if t==True:
        if t==True:

            # LOAD TRAIN/VALID DATASETS, 
            if params["img_generator"]==None or model_unit_test==True:
                "use preset value that donwload all images form the dataset as they are, with defaul size"
                X, batch_labels = load_raw_img_batch_with_custom_datagen(                                                
                    path=PATH_raw, 
                    subset_names=train_subset_names,  
                    n_next_datagen = 1, 
                    ImageDataGenerator_kwargs = None,
                    datagen__target_size=params["img_size"],    
                    subset_batch_size = None, # None == all samples in the file will be loaded, once,                                           
                    shuffle_batch = False,  # 
                    shuffle_all = False,                     
                    verbose=verbose
                    )

            else:
                X, batch_labels = load_raw_img_batch_with_custom_datagen(                                                
                    path=PATH_raw, 
                    subset_names=train_subset_names,  
                    n_next_datagen = params["img_generator"]["n_next_datagen"], 
                    ImageDataGenerator_kwargs = params["img_generator"]["ImageDataGenerator_kwargs"],
                    datagen__target_size = params["img_size"],    
                    subset_batch_size = None, # None == all samples in the file will be loaded, once,                                           
                    shuffle_batch = False,  # 
                    shuffle_all = params["img_generator"]["shuffle_all"],                     
                    verbose=verbose
                    )           
                
            # define class_encoding/decoding
            train_classes  = list(batch_labels.classname.unique())
            class_encoding = dict(zip(train_classes,list(range(len(train_classes)))))
            class_decoding = dict(zip(list(list(class_encoding.values())), list(class_encoding.keys()))) # reverse on class_encoding,

            # adapt matrices for making predictions with the model, 
            X = X.astype(np.float)
            y = pd.Series(batch_labels.classname).map(class_encoding).values.astype("int")
            
            # Create train/valid/test sets
            if model_unit_test==False:
            
                if valid_subset_names==None:
                    # -1- Create train/test sets
                    _, _, idx_tr, idx_valid = train_test_split(
                        X, np.arange(X.shape[0], dtype="int"), 
                        train_size=params["train_test_split__train_size"], 
                        #test_size=(1-params["train_test_split__train_size"]),
                        random_state=random_state_nr    # Caution, random_state_nr must be the same as in the above, 
                        )
                    # .. separate train and valid dataset
                    X_tr = X[idx_tr] 
                    X_valid = X[idx_valid]
                    y_tr = y[idx_tr] 
                    y_valid = y[idx_valid]
                    batch_labels_tr = batch_labels.iloc[idx_tr,:] 
                    batch_labels_valid = batch_labels.iloc[idx_tr,:] 
                else:
                    # .. adapt naming for train set, 
                    X_tr = X
                    y_tr = y
                    batch_labels_tr = batch_labels
                    idx_tr = np.arange(batch_labels_tr.shape[0])
                    
                    if augment_valid_data==False or params["img_generator"]==None:
                        # .. load valid dataset independely, with no augomenation, 
                        X_valid, batch_labels_valid = load_raw_img_batch_with_custom_datagen(                                                
                            path=PATH_raw, 
                            subset_names=valid_subset_names,  
                            n_next_datagen = 1, 
                            ImageDataGenerator_kwargs = None,
                            datagen__target_size=params["img_size"],    
                            subset_batch_size = None, # None == all samples in the file will be loaded, once,                                           
                            shuffle_batch = True,  # 
                            shuffle_all = True,                     
                            verbose=verbose
                            )
                    else:
                        "use the same augmentation as with train dataset"
                        X_valid, batch_labels_valid = load_raw_img_batch_with_custom_datagen(                                                
                            path=PATH_raw, 
                            subset_names=valid_subset_names,  
                            n_next_datagen = params["img_generator"]["n_next_datagen"], 
                            ImageDataGenerator_kwargs = params["img_generator"]["ImageDataGenerator_kwargs"],
                            datagen__target_size = params["img_size"],    
                            subset_batch_size = None, # None == all samples in the file will be loaded, once,                                           
                            shuffle_batch = False,  # 
                            shuffle_all = params["img_generator"]["shuffle_all"],                     
                            verbose=verbose
                            )                      
                    # finally, adapt matrices with valid data for making predictions with the model, 
                    X_valid = X_valid.astype(np.float)
                    y_valid = pd.Series(batch_labels_valid.classname).map(class_encoding).values.astype("int")       
                    idx_valid = np.arange(batch_labels_valid.shape[0])

                # -2- Load and create test set, 
                if test_subset_names==None:
                    "test data are not available at this stage only rain and valid datsets are ussed further on"
                    # assign all datasets to dict,
                    Xy_names = ["train", "valid"]
                    X_dct = dict(zip(Xy_names, [X_tr, X_valid]))
                    y_dct = dict(zip(Xy_names, [y_tr, y_valid]))
                    idx_y_dct = dict(zip(Xy_names, [idx_tr, idx_valid])) 
                    batch_labels_dct = dict(zip(Xy_names, [batch_labels_tr, batch_labels_valid]))  
                    # ....
                    if verbose==True:
                        print("CAUTION ! . test_subset_names==None, the grid will not calulate test score for fitted models, only validation accuracy, ")
                    else:
                        pass
                
                if test_subset_names!=None:
                    "load test datset ass it ism, with no imag augmentation or shuffling"
                    X_te, batch_labels_te = load_raw_img_batch_with_custom_datagen(                                                
                        path=PATH_raw, 
                        subset_names=test_subset_names,  
                        n_next_datagen = 1, 
                        ImageDataGenerator_kwargs = None,
                        datagen__target_size=params["img_size"],    
                        subset_batch_size = None, # None == all samples in the file will be loaded, once,                                           
                        shuffle_batch = False,  # 
                        shuffle_all = False,                     
                        verbose=verbose
                        )
                    # adapt matrices for making predictions with the model, 
                    X_te = X_te.astype(np.float)
                    y_te = pd.Series(batch_labels_te.classname).map(class_encoding).values.astype("int")
                    
                    # assign all datasets to dict,
                    Xy_names = ["train", "valid", "test"]
                    X_dct = dict(zip(Xy_names, [X_tr, X_valid, X_te]))
                    y_dct = dict(zip(Xy_names, [y_tr, y_valid, y_te]))
                    idx_y_dct = dict(zip(Xy_names, [idx_tr, idx_valid, np.arange(batch_labels_te.shape[0])])) 
                    batch_labels_dct = dict(zip(Xy_names, [batch_labels_tr, batch_labels_valid, batch_labels_te]))
                   
         
            # .. cd .. Create train/valid/test sets for model unit test 
            if model_unit_test==True:
                "if True, all data subsets are exactly the same ! -> acc shoudl be = 1 after 40-50 epoch, or more in case of larger datasets"
                X_dct = dict()
                y_dct = dict()
                idx_y_dct = dict()
                batch_labels_dct = dict()
                X_tr = X; X_valid=X; X_te = X
                y_tr = y; y_valid=y; y_te = y
                batch_labels_tr = batch_labels; batch_labels_valid=batch_labels; batch_labels_te = batch_labels

                Xy_names = ["train", "valid", "test"]
                for xy_name in Xy_names:
                    X_dct[xy_name] = X
                    y_dct[xy_name] = y
                    idx_y_dct[xy_name] = np.arange(batch_labels.shape[0])
                    batch_labels_dct[xy_name] = batch_labels.copy()
                # ...
                if verbose==True:
                    print("CAUTION ! . model_unit_test==True, all datasets (train, valid and test) are the same, expected acc=1, for all of them")
                else:
                    pass
            
            # ....
            if verbose==True:
                print(".. FOLLOWIING DATASETS WERE LOADED:")
                for xy_name in Xy_names:
                    print(f"{xy_name}: X: {X_dct[xy_name].shape}, y: {y_dct[xy_name].shape}, batch_labels: {batch_labels_dct[xy_name].shape}")
            else:
                pass
    
    # return quite a lot, 
    return (X_dct, y_dct, batch_labels_dct, idx_y_dct, class_encoding,  class_decoding) 


