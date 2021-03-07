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
from src.utils.image_augmentation import * # some basic tools create for image augmentastion using keras tools, here used for building batch_labels table,  





# Function, .................................................................................................................

def encode_images(*, 
    # .. dastaset name & directories, 
    dataset_name=None,# dataset name used when saving encoded files, logfiles and other things, related to encoding, 
    load_dir=None,   # full path to input data, ie. file folder with either folders with images names after class names, or folders with subsetnames, and folders names after each class in them, 
    subset_names=None,# list, ust names of files in the load_dir, if any, 
    save_dir=None, # all new files, will be saved as one batch, with logfile, if None, load_dir will be used, 

    # .. encoding module parameters, 
    module_name=None, 
    module_location, # full path to a given module,
    img_target_size, # image resolution in pixels, 
    generator_batch_size =1000, 
    generator_shuffle    =False, 
        
    # .. other, 
    save_files=True,
    verbose=True                             
):
    """
        Function does the following:
        - extracts features from rgb figures, 
        - saves each batch in npy file format, [img_nr, x_pixel, y_pixel, 3]
        - saves csv file with info on each image in that file, 
        - creates log file with info on files with images that were encoded and 
        
        # Arguments
        . datase_tname          : str, an arbitrary name used at the beginning of all files created
        . module_name          : str  eg: "mobilenet", used as above in each file create with that function, 
        . module_location      : str, either ulr to tfhub module used for feature extraction,
                                      or path to directory with locally saved module, 
        . load_dir             : str, path to directory that contains the folder with folders containgn classes, 
        . save_dir             : str, path to directory whther all files will be stored, 
        . folders_in_load_dir  : list with strings, each string is an exact folder name, that contains images stored in folders with class names, 
        . img_target_size      : tuple, (int, int), img (height, width) size in pixesl, Caution, this functions works only with RGB FILES, 
                                 thus, the final, batch dimensions are [?, height, width, 3]
        . generator_batch_size : int, value for keras, ImageDataGenerator.flow_from_directory, batch size, 
                                 the generator will use only max available images, if less is in the dataset,
                                 or only the first 10000 or other requested number, if more is available, 
                                 be carefull to set generator_shuffle = True, while using a subset of images to have all classes
                                 represented, 
        . generator_shuffle    : bool, value for keras, ImageDataGenerator.flow_from_directory,                  
        . ...
        . verbose              : bool, default==False, 
        . save_files           : bool, if True, encoded batch, logfile, and labels, will be saved ib save_dir, 
                                       if False, encoded batch and labels will be returned as dictionaries in list
                                       with folders_in_load_dir as keys, no log file will be created, 
        # Returns
        . log file             : pd.dataFrame, with basic info on each dataset encoded, and names used to save files for it, 
        
        # saves
        . log file             : pd.dataFrame, saved in save_dir as .csv
        . img_batch_info       : pd.dataFrame, saved in save_dir as .csv
        . img_batch_features   : .npy with [(images found in a given set), feture_number], with encoded features from the given module, 
        
        # example - use case, 

        # ... from, 
        module_path =  os.path.join(basedir,"models/imagenet_mobilenet_v2_100_224_feature_vector_2") # path to local directory, 
        module_path = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2' # as url
        # ...
        module_name = "mobilenet"
        module_img_imput_size = (224, 224)  
        # ....
        encodedimageges_dct, lables_dct = encode_images_with_tfhubmodule( 
                    datasetname="augmented_swissroads", 
                    module_name=module_name, 
                    module_location=module_path, 
                    load_dir= os.path.join(basedir, "data/interim/augmented_swissroads/"), 
                    folders_in_load_dir=["test"], 
                    save_dir=os.path.join(basedir, "data/interim/augmented_swissroads/"),
                    img_target_size=module_img_imput_size,                     
                    save_files=True,
                    verbose=True
            )
           
    """

    # ...........................................................................
    # set up, 
    
    if load_dir==None:
        load_dir = os.path.dirname(os.getcwd())
    else:
        pass
    
    if save_dir==None:
        save_dir = load_dir
    else:
        pass
        
    if module_name==None:
        module_name = "tfmodule"
    else:
        pass
    
    if dataset_name==None:
        dataset_name = "encodeddata"
    else:
        pass
    
    # create save directory, if not available, 
    try:
        os.mkdir(save_dir)
        if verbose==True:
            print(f"Created: {save_dir}")
    except: 
        pass
    
                
    
    # ........................................................................
    # Create iterators for each dataset, 
    
    # Create Image generator, with rescaling, - I assumed we will wirk on rgb pictures, 
    datagen = ImageDataGenerator(rescale=1/255)

    #.. collect iterators for each subset, if available, and place them in dictionary, 
    iterators_dct = dict()
    
    # .. - there is more then one subset of the data, eg validation and test and train data 
    if subset_names!=None:
        for setname in subset_names:    
            iterators_dct[setname] = datagen.flow_from_directory(
                os.path.join(load_dir, setname), # each subset is loaded separately, 
                batch_size=generator_batch_size, # it will use only max available pictures in a given folder,  
                target_size=img_target_size, 
                shuffle=generator_shuffle     # here I use generator only to explote the data
                )
    else:
        iterators_dct[dataset_name] = datagen.flow_from_directory(
                load_dir,
                batch_size=generator_batch_size, # it will use only max available pictures in a given folder,  
                target_size=img_target_size, 
                shuffle=generator_shuffle     # here I use generator only to explote the data
                )
        
    # ..
    if verbose==True: 
        print(f"\n\n{''.join(['.']*80)}\n Creating DataGenerators for: {dataset_name}; {module_name};\n{''.join(['.']*80)}\n")
    else:
        pass    
    
    
    
    # .......................................................................    
    # finally, in case there are no subsets, use the same name for subset as for dataset in all subsequent datasets, 
    if subset_names==None:
        subset_names = [dataset_name]
    else:
        pass    
    "important - it must be done here !"
    
    
    
    # .......................................................................  
    # Create tf graph,
    img_graph = tf.Graph()

    with img_graph.as_default():

        # load the module, by default i set TF 1.x models, and if they dont work, I assumed that you will load TF 2.X modules with hub.load() functions used as below
        try:
            feature_extractor = hub.Module(module_location, trainable=False) # tf1 modules, tahes a bit of time, to check if there is an error, but it was the fasted I could use with coding, 
        except:
            feature_extractor = hub.KerasLayer(module_location) # tf2 modules,
            
        # Create input placeholder for imput data, 
        input_imgs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, img_target_size[0], img_target_size[1], 3])

        # A node with the extracted high-level features
        imgs_features = feature_extractor(input_imgs)

        # Collect initializers
        init_op = tf.group([                              # groups different type of initializers into one init_op
            tf.compat.v1.global_variables_initializer(),  # used for almost all tf graphs, 
            tf.compat.v1.tables_initializer()             # speciffic for tfhub
        ])

    img_graph.finalize() # Good practice: make the graph "read-only"

    #.. create and initialize the session, 
    "following instruction on: https://www.tensorflow.org/hub/common_issues"
    sess = tf.compat.v1.Session(graph=img_graph)
    sess.run(init_op)    
    
    #.. info on tf graph, 
    if verbose==True:
        print(f"\n{''.join(['.']*80)}\n TF Graph;")
        print(f"Feature extraction Module, from: {module_location}")
        #print("Signatures: ",feature_extractor.get_signature_names()) # eg: ['default', 'image_feature_vector'] - signatures, can be used to extract subset of NN layers
        #print("Expected Imput shape, dtype: ", feature_extractor.get_input_info_dict()) # eg: {'images': <hub.ParsedTensorInfo shape=(?, 224, 224, 3) dtype=float32 is_sparse=False>}
        #print("Expected Output shape, dtype: ",feature_extractor.get_output_info_dict()) # eg: {'default': <hub.ParsedTensorInfo shape=(?, 1280) dtype=float32 is_sparse=False>}

        
        
        
    # .......................................................................   
    # Use iterators, and tf graph, extract high level features,
    "extract features and labels with each batch, and save as npz, file"
    
    # objects to store new before saving or returning, 
    file_sets = list()
    encoded_batch_dict = dict()
    batch_labels_dict = dict()
    
    # ..
    for setname in subset_names:
              
        # img batch encoding, ...............................................
        #.. Load batch,
        img_batch, img_labels = next(iterators_dct[setname])

        #.. Extract features from the batch of images, 
        img_batch_features = sess.run(imgs_features, feed_dict={input_imgs: img_batch})
        
        #.. add to dict
        encoded_batch_dict[setname] = img_batch_features 
        
        # create human-friendly labels, ..................................... 
        'prepare info on each image'
        to_forget, img_batch_info = create_augmented_images(external_generator=iterators_dct[setname], augm_img_nr=0) # used only to get info df, 
        
        #.. add to dict
        batch_labels_dict[setname] = img_batch_info
        
        # info
        if verbose==True: 
            print(f"{''.join(['.']*80)}\n Ecoding imgages in one batch for < {setname} > dataset;")
            print(f"Feature number = {img_batch_features.shape}")
            print(f"label table shape = {img_batch_info.shape}")        
        

        
        # save files in save_dir, ...........................................
        if save_files==True:

            # save img_batch_features as .npy file
            os.chdir(save_dir)
            encoded_img_file_name = f"{module_name}_{dataset_name}_{setname}_encoded.npy"
            np.save(encoded_img_file_name , img_batch_features)

            # save labels as csv, 
            batch_label_file_name = f"{module_name}_{dataset_name}_{setname}_labels.csv"
            img_batch_info.to_csv(batch_label_file_name, header=True, index=False)
            
            # info on the above, 
            if verbose==True: 
                print(f"Saved as:\n{encoded_img_file_name} and {batch_label_file_name}")
                print(f"saved in:\n{os.getcwd()}")        

            # create log file required for easy loading of the batch and lebel files,   
            file_sets.append({
                "module_name": module_name,
                "datasetname": setname,
                "img_batch_features_filename": encoded_img_file_name,
                "img_batch_info_filename": batch_label_file_name,
                "batch_size": img_batch_info.shape[0],
                "created_dt":pd.to_datetime("now"),
                "module_source": module_location
            }) 
            
            
    # .......................................................................             
    # Save log table for entire datset, with all subsets in different rows,  
    if save_files==True:
        
        os.chdir(save_dir)
        summary_table_filename = f"{module_name}_{dataset_name}_logfile.csv"
        summary_table = pd.DataFrame(file_sets)
        summary_table.to_csv(summary_table_filename, header=True, index=False)

        #.. info
        if verbose==True: 
            print(f"{''.join(['.']*80)}\n Creating logfile for < {dataset_name} >;") 
            print(f"saved as:  {summary_table_filename}") 
            print(f"in: {save_dir}") 
  
    
    # return the batch and labels, just in case, 
    return encoded_batch_dict, batch_labels_dict


  




# Function, .....................................................................................................

def encode_images_with_tfhubmodule(*, 
    datasetname, 
    module_name, 
    module_location, 
    load_dir, 
    folders_in_load_dir, 
    save_dir,
    img_target_size, 
    save_files=True,
    verbose=False
):
    """
  
  
        Function does the following:
        - extracts features from rgb figures, 
        - saves each batch in npy file format, [img_nr, x_pixel, y_pixel, 3]
        - saves csv file with info on each image in that file, 
        - creates log file with info on files with images that were encoded and 
        
        caution, this is my older function, (Oct 2018), that I kept for back compatibility, 
        
        
        # Arguments
        . datasetname          : str, an arbitrary name used at the beginning of all files created
        . module_name          : str  eg: "mobilenet", used as above in each file create with that function, 
        . module_location      : str, either ulr to tfhub module used for feature extraction,
                                      or path to directory with locally saved module, 
        . load_dir             : str, path to directory that contains the folder with folders containgn classes, 
        . save_dir             : str, path to directory whther all files will be stored, 
        . folders_in_load_dir  : list with strings, each string is an exact folder name, that contains images stored in folders with class names, 
        . img_target_size      : tuple, (int, int), img (height, width) size in pixesl, Caution, this functions works only with RGB FILES, 
                                thus, the final, batch dimensions are [?, height, width, 3]
        . verbose              : bool, default==False, 
        . save_files           : bool, if True, encoded batch, logfile, and labels, will be saved ib save_dir, 
                                       if False, encoded batch and labels will be returned as dictionaries in list
                                       with folders_in_load_dir as keys, no log file will be created, 
        # Returns
        . log file             : pd.dataFrame, with basic info on each dataset encoded, and names used to save files for it, 
        
        # saves
        . log file             : pd.dataFrame, saved in save_dir as .csv
        . img_batch_info       : pd.dataFrame, saved in save_dir as .csv
        . img_batch_features   : .npy with [(images found in a given set), feture_number], with encoded features from the given module, 
        
        # example - use case, 

        # ... from, 
        module_path =  os.path.join(basedir,"models/imagenet_mobilenet_v2_100_224_feature_vector_2") # path to local directory, 
        module_path = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2' # as url
        # ...
        module_name = "mobilenet"
        module_img_imput_size = (224, 224)  
        # ....
        encodedimageges_dct, lables_dct = encode_images_with_tfhubmodule( 
                    datasetname="augmented_swissroads", 
                    module_name=module_name, 
                    module_location=module_path, 
                    load_dir= os.path.join(basedir, "data/interim/augmented_swissroads/"), 
                    folders_in_load_dir=["test"], 
                    save_dir=os.path.join(basedir, "data/interim/augmented_swissroads/"),
                    img_target_size=module_img_imput_size,                     
                    save_files=True,
                    verbose=True
            )
           
    """

    #### ........................................................................
    #### Step 0. Create iterators for each dataset, 
    
    #.. Create Image generator, with rescaling, - I assumed we will wirk on rgb pictures, 
    datagen = ImageDataGenerator(rescale=1/255)

    #.. collect iterators for each datatype in load_dir
    if verbose==True: print(f"\n\n{''.join(['.']*80)}\n Creating DataGenerators for: {datasetname}; {module_name};\n{''.join(['.']*80)}\n")
    iterators_dct = dict()
    for setname in folders_in_load_dir:    
        iterators_dct[setname] = datagen.flow_from_directory(
            os.path.join(load_dir, setname),
            batch_size=10000000000, # it will use only max available pictures in a given folder,  
            target_size=img_target_size, 
            shuffle=False     # here I use generator only to explote the data
            )
       
    #### .......................................................................  
    #### Step 1. create tf graph,
    img_graph = tf.Graph()

    with img_graph.as_default():

        # load the module, by default i set TF 1.x models, and if they dont work, I assumed that you will load TF 2.X modules with hub.load() functions used as below
        try:
            feature_extractor = hub.Module(module_location, trainable=False) # tf1 modules, tahes a bit of time, to check if there is an error, but it was the fasted I could use with coding, 
        except:
            feature_extractor = hub.KerasLayer(module_location) # tf2 modules,
            
        # Create input placeholder for imput data, 
        input_imgs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, img_target_size[0], img_target_size[1], 3])

        # A node with the extracted high-level features
        imgs_features = feature_extractor(input_imgs)

        # Collect initializers
        init_op = tf.group([                    # groups different type of initializers into one init_op
            tf.compat.v1.global_variables_initializer(),  # used for almost all tf graphs, 
            tf.compat.v1.tables_initializer()             # speciffic for tfhub
        ])

    img_graph.finalize() # Good practice: make the graph "read-only"

    #.. create and initialize the session, 
    "following instruction on: https://www.tensorflow.org/hub/common_issues"
    sess = tf.compat.v1.Session(graph=img_graph)
    sess.run(init_op)    
    
    #.. info on tf graph, 
    if verbose==True:
        print(f"\n{''.join(['.']*80)}\n TF Graph;")
        print("Feature extraction Module")
        print(f"from: {module_location}")
        #print("Signatures: ",feature_extractor.get_signature_names()) # eg: ['default', 'image_feature_vector'] - signatures, can be used to extract subset of NN layers
        #print("Expected Imput shape, dtype: ", feature_extractor.get_input_info_dict()) # eg: {'images': <hub.ParsedTensorInfo shape=(?, 224, 224, 3) dtype=float32 is_sparse=False>}
        #print("Expected Output shape, dtype: ",feature_extractor.get_output_info_dict()) # eg: {'default': <hub.ParsedTensorInfo shape=(?, 1280) dtype=float32 is_sparse=False>}

    
    #### .......................................................................    
    #### Step 2. Using iterators, and tf graph, extract high level features,  
    "extract features and labels with each batch, and save as npz, file"
    
    # list to store log files, 
    file_sets = list()
    encoded_batch_dict = dict()
    batch_labels_dict = dict()
    
    
    # iterate over each dataset name in a given folder, 
    for setname in folders_in_load_dir:
          
            
        # -1- img batch encoding,  
        
        #.. Load batch,
        img_batch, img_labels = next(iterators_dct[setname])

        #.. Extract features from the batch of images, 
        img_batch_features = sess.run(imgs_features, feed_dict={input_imgs: img_batch})
        
        #.. add to dict
        encoded_batch_dict[setname] = img_batch_features

        
        # -2- img friendly labels in pd.dataframe,  
            
        #.. prepare info on each image,
        to_forget, img_batch_info = create_augmented_images(external_generator=iterators_dct[setname],augm_img_nr=0) # used only to get info df, 
        tempdf = img_batch_info.imgname.str.split(pat="_", expand=True)
        img_batch_info.imgname = tempdf.iloc[:,1]
        img_batch_info.imgtype = tempdf.iloc[:,0]
        
        #.. add to dict
        batch_labels_dict[setname] = img_batch_info
        
        
        
        # - - info on 1 and 2,
        
        if verbose==True: 
            print(f"{''.join(['.']*80)}\n Ecoding imgages in one batch for < {setname} > dataset;")
            print(f"Feature number = {img_batch_features.shape}")
            print(f"label table shape = {img_batch_info.shape}")        
        
        
        # -3- saving encoded batch and label files in save_dir
        if save_files==True:

            # save img_batch_features as .npy file
            os.chdir(save_dir)
            file_name = f"{module_name}_{datasetname}_{setname}_encoded.npy"
            np.save(file_name, img_batch_features)

            # save labels as csv, 
            second_file_name = f"{module_name}_{datasetname}_{setname}_labels.csv"
            img_batch_info.to_csv(second_file_name, header=True, index=False)
            
            # info on the above, 
            if verbose==True: 
                print(f"Saved as:\n{file_name} and {second_file_name}")
                print(f"saved in:\n{os.getcwd()}")        

            # create log file required for easy loading of the batch and lebel files,   
            file_sets.append({
                "module_name": module_name,
                "datasetname": setname,
                "img_batch_features_filename": file_name,
                "img_batch_info_filename": second_file_name,
                "batch_size": img_batch_info.shape[0],
                "created_dt":pd.to_datetime("now"),
                "module_source": module_location
            }) 
            
            
    #### Step 3. Save log table with the summary,     
        
    if save_files==True:
        
        os.chdir(save_dir)
        summary_table_filename = f"{module_name}_{datasetname}_logfile.csv"
        summary_table = pd.DataFrame(file_sets)
        summary_table.to_csv(summary_table_filename, header=True, index=False)

        #.. info
        if verbose==True: 
            print(f"{''.join(['.']*80)}\n Creating logfile for < {datasetname} >;") 
            print(f"saved as:  {summary_table_filename}") 
            print(f"in: {save_dir}") 
  
    #### return the batch and labels, just in case, 
    return encoded_batch_dict, batch_labels_dict