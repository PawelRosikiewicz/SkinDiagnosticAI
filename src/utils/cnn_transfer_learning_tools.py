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

import cv2
import numpy as np # support for multi-dimensional arrays and matrices
import pandas as pd # library for data manipulation and analysis
import seaborn as sns # advance plots, for statistics, 
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt # for making plots, 

from PIL import Image, ImageDraw
import matplotlib.gridspec
from scipy.spatial import distance
from scipy.cluster import hierarchy
from matplotlib.font_manager import FontProperties
from scipy.cluster.hierarchy import leaves_list, ClusterNode, leaders
from sklearn.metrics import accuracy_score

import graphviz # allows visualizing decision trees,
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier # accepts only numerical data
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

#. require for plots below, 
from src.utils.image_augmentation import * # to create batch_labels files, 
from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch
from src.utils.tools_for_plots import create_class_colors_dict
from src.utils.example_plots_after_clustering import plot_img_examples, create_spaces_between_img_clusters, plot_img_examples_from_dendrogram
from src.utils.annotated_pie_charts import annotated_pie_chart_with_class_and_group, prepare_img_classname_and_groupname


#############################################
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import Sequential, activations, initializers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#.  os.environ['KMP_DUPLICATE_LIB_OK']='True'  # To avoid restaring the kernel with keras, preffered solution; use conda install nomkl
#############################################





# Fuction ...................................................................................

def create_keras_two_layer_dense_model(*,
    input_size, 
    output_size, 
    verbose=False,
    **kwargs
                                       
):

    # ...................................................
    # Create model
    model = Sequential()
    
    #.. First hidden layer
    model.add(Dense(
        input_dim=input_size,
        units=kwargs['h1_unit_size'], 
        activation=kwargs["h1_activation"],
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=initializers.VarianceScaling(scale=2.0, seed=0)
    ))
    model.add(tf.keras.layers.Dropout(kwargs["h1_Dropout"]))
    

    #.. Output layer
    model.add(Dense( 
        units=output_size, 
        activation=kwargs["out_activation"],
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=initializers.VarianceScaling(scale=1.0, seed=0)
    ))

    # Print network summary
    if verbose==True:
        print(model.summary())
    else:
        pass

    # ...................................................
    # Define Loss Function and Trianing Operation 
    """ # [option]: Use only default values,  
        model.compile( optimizer='sgd', 
            loss='sparse_categorical_crossentropy', 
            metrics=['acc'])
    """
    model.compile(
        optimizer= kwargs["optimizer"],
        loss= losses.sparse_categorical_crossentropy,
        metrics= kwargs["metrics"] # even one arg must be in the list
    )
    
    return model

    

    
    
# Function, ................................................................................
def plot_NN_loss_acc(*, model_history_df, title="kbdkgkd", n_mean=3, figsize=(8,4), top=0.75):
    ''' 
        small function to plot loss and accuracy over epoch using data created with history.history() keras functions, 
        the columns shodul be called acc, loss, and val_acc, val_loss, 
        # ...
        . model_history_df   : dataframe, created with history.history() keras functions (see in the above)
        . n_mean             : int, how many last results use to caulate values displayed in suplot title
        . title.             : str, plot title, 
    '''
    #.. figure, axes, 
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(title)

    #.. Plot accuracy values
    ax1.plot(model_history_df.loc[:,'loss'], label='train loss')
    ax1.plot(model_history_df.loc[:,'val_loss'], label='val loss')
    ax1.set_title('Mean validation loss {:.3f}'.format(np.mean(model_history_df.loc[:, 'val_loss'][-n_mean:])))
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss value')
    ax1.grid(ls="--", color="grey")
    ax1.legend()

    #.. Plot accuracy values
    ax2.plot(model_history_df.loc[:, 'acc'], label='train acc')
    ax2.plot(model_history_df.loc[:, 'val_acc'], label='val acc')
    ax2.set_title('Mean validation acc {:.3f}'.format(
    np.mean(model_history_df.loc[:, 'val_acc'][-n_mean:])))
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_ylim(0,1)
    ax2.grid(ls="--", color="grey")
    ax2.legend()
    
    # ... 
    plt.tight_layout()
    plt.subplots_adjust(top=top)
    plt.show()
        
        
        
        
        
        
        
# Fuction, ........................................................................
# new 2020.12.14
def CNN_GridSearch(*,

    # ... models and manes, 
    method, 
    grid,
    run_name="",
    dataset_name="",
    dataset_variant="",
    module_name="",
    
    # .... input data names and paths
    file_namepath_table,
    file_namepath_table_dict=None,
    PATH_batch_data=None,             # if None, it is provided by the file
    PATH_batch_labels=None,           # if None, it is provided by the file                    
    PATH_results=None,                # If None, the same as the last PATH_features will be used,
    
    # .... names used to search for subset names and save results
    class_encoding,
    class_decoding,
    dropout_value='to_dropout',
    train_subset_name="train",        # because I donth have to call that train in my files, 
    valid_subset_name="valid",        # if None, train_proportion will be used
    test_subset_name_list="test",     # if None, the loist is simply shorter, 
    train_proportion=0.7,             # used only if subset_names_valid = None, by none, 
    unit_test=True,
    
    # ... results and info, 
    store_predictions=True,
    track_progres=True,   
    verbose=False,
                   
    # info speciffic to cnn methods     
    plot_history=True,               # applied only if verbose==True
    model_fit__verbose=1
):
    '''
       Custom function to run grid search on fully connected models 
       with features extracted from images as input,

       # Inputs, 
  
        # ... model description, 
        . method              : str, model descrition, 
        . grid                : str, model descrition, 
        . run_name            : str, model descrition, 
        . dataset_name        : str, model descrition, 
        . dataset_variant     : str, model descrition, 
        . module_name         : str, model descrition, 

        # .... input data names and paths
        . file_namepath_table,                : dataframe, 
        . file_namepath_table_dict=None       : contains colnames for file names/path 
                                                  to feastures and labels in file_namepath_table
        # .... names used to search for subset names and save results
        class_encoding                        : dict key=class label, value int, 
        class_decoding                        : dict, opposite to encoding
        dropout_value                         : str, a value used to remove samples of classes form the input data
        train_subset_name                     : str, eg train, shoudl correspond to subset_name in  file_namepath_table
        valid_subset_name                     : str, eg valid, shoudl correspond to subset_name in  file_namepath_table
        test_subset_name_list                 : str, list or None, eg test, [test 1 and test 2]
        unit_test                             : if True, train datasets used as valid and test, 

        PATH_batch_data=None                  : if None, it is provided by the file
        PATH_batch_labels=None                : if None, it is provided by the file 

        # ... results and info, 
        store_predictions                     : bool, if Truze, predictions and lost of other info storeed
        track_progres                         : bool, small info on process running printed
        verbose                               : bool, if True, full info displayed
        plot_history                          : bool, if True plots acc and loss from model history
        model_fit__verbose                    : 0,1,2, for versbose in keras sequencial model fit, 
    '''

    # ......................................................................    
    # Set up, 
    # ......................................................................    

    # - contats
    colname_with_classname_in_batch_labels_table = "classname"
    
    # - variables,
    model_ID           = -1   # to start usnique counts of the models from 0 
    batchfile_table    = file_namepath_table # legacy issue
    available_subsets_in_batchfile_table = pd.Series(batchfile_table.subset_name.tolist())
    
    # - to store the results, 
    model_acc_and_parameters_list = list()
    model_predictions_dict        = dict()
    model_parameters_dict         = dict() # for iterations from grid
    model_history_dict            = dict()
    
    # - dict to use values from file_namepath_table to load the data
    if file_namepath_table_dict==None:
        file_namepath_table_dict = {
            "subset_name":'subset_name',
            "batch_data_file_name":'extracted_features_file_name',
            "batch_data_file_path":'extracted_features_file_path',
            "batch_labels_file_name":'labels_file_name',
            "batch_labels_file_path":'labels_file_path'
        }
    else:
        pass
    
    # - check if samples shdoul be dropped
    samples_to_drop_detected=(np.array(list(class_encoding.keys()))==dropout_value).sum()>0

    

    # ......................................................................    
    # SELECT DATASET NAMES 
    # ......................................................................    
    '''
        Create xy_names with subset names used as train valid, test     
        xy names will be used to find names oF datafiles to load 
        in the batchfile_table
    '''           
    # 1. train set (train) - 1st POSITION IN xy_names -
    xy_names = []
    xy_names.append(train_subset_name)     # 1st is always the train dataset,
    # check if you are asking for existing subset name
    if sum(available_subsets_in_batchfile_table==train_subset_name)>=1:
        requested_subsets_are_avaialble_in_batchfile_table = True
    else:
        requested_subsets_are_avaialble_in_batchfile_table = False
        
    # ......................................................................
    if unit_test == True:
        xy_names.append(train_subset_name) # later on updated to valid, 
        xy_names.append(train_subset_name) # later on updatef to train
        some_values_required_to_build_valid_dataset_are_missing=False # because it is not used at all
    else:
        
        # 2. validation set (valid), - 2nd POSITION IN xy_names -
        if valid_subset_name!=None:  
            'valid subset specified externally'
            xy_names.append(valid_subset_name) # second is always the valid dataset,
            train_proportion=None              # done here, to ensure it will be ignored later on, 
            # check if you are asking for existing subset name
            if sum(available_subsets_in_batchfile_table==valid_subset_name)==0:
                requested_subsets_are_avaialble_in_batchfile_table = False
            else:
                pass
        else: 
            'valid subset will be created form train set with specific proportion'
            xy_names.append("valid") # second is always the valid dataset,
            # Ensure we have that value as flot betwe 0 and 1
            try:
                train_proportion = float(train_proportion)
            except:
                if verbose==True:
                    print(f"ERROR: train_proportion or valid_subset_name are missing !")
                else:
                    pass
        # test if you even can find validation dataset and stop the system form loading data and gridsearchibng        
        if  train_proportion==None and valid_subset_name==None:
            some_values_required_to_build_valid_dataset_are_missing = True
        else:
            some_values_required_to_build_valid_dataset_are_missing = False
        
            
        # 3. test sets (test)
        'can be more then one test set, but they must have different names eg test1, test2, ... '
        if test_subset_name_list!=None: 
            # adapt to loo below, it the string was given with just one name, 
            if isinstance(test_subset_name_list, str):
                test_subset_name_list = [test_subset_name_list]
            
            # place each test names, but only if it is in batchfile_table,
            for test_subset_name in test_subset_name_list:
                if sum(available_subsets_in_batchfile_table==test_subset_name)==0:
                    'to check if you are asking for existing subset name or there is some problem'
                    requested_subsets_are_avaialble_in_batchfile_table = False
                else:              
                    xy_names.append(test_subset_name)
        else: 
            pass # xy_names will be simply shorter, and no     
    
    # check if all values in xy_names are unique with exception of unit test
    'otherwise loaded data will be oiverwritten in dict, and you will use them without knowing that these are same datsets'
    if unit_test==True:
        all_subsetnames_in_xy_names_are_unique = True # its not true, but I will modify it after loading the data in that single case
    else:
        all_subsetnames_in_xy_names_are_unique = len(pd.Series(xy_names).unique().tolist())==len(xy_names)
    

    
    # ......................................................................    
    # STOP POINT AFTER SELECTING AND COMBINING DATASET NAMES
    # ......................................................................    
    '''
        this part of the code will stop any further actions, because at least one element 
        of information provided for dataset to load, was not correct, and it could corrupt 
        the results, 
    '''
    if requested_subsets_are_avaialble_in_batchfile_table==False:
        if verbose==True or track_progres==True:
            print("KeyError: at least one subset name is different then subset names used in batchfile_table or was not there") 
            print("the operations were stopped")
        pass
    elif some_values_required_to_build_valid_dataset_are_missing==True:
        if verbose==True or track_progres==True:
            print("KeyError: train_proportion or valid_subset_name are missing") 
            print("the operations were stopped")
        pass      
    elif all_subsetnames_in_xy_names_are_unique==False:
        if verbose==True or track_progres==True:
            print("KeyError: at least one subset name is not unique - please make sure they are unique")
            print("the operations were stopped")
        pass           
    else:
        
         
        # ......................................................................    
        # GRID SEARCH
        # ......................................................................  
        '''
            here the data will be loaded and module constructed, and predictions made
        '''
        
        if track_progres==True or verbose==True:
            print(f"\nGrid search for - {method} - with {len(grid)} params combinations: {pd.to_datetime('now')}")
            print(f" method: {method}")
            print(f"run_name: {run_name}")
            print(f"dataset_name: {dataset_name}")
            print(f"dataset_variant: {dataset_variant}")
            print(f"module_name: {module_name}")
            print(f"Number of combinations: {len(grid)}")
            print(f"Unit test run: {unit_test}")
            print("")
        else: 
            pass 
        # ... 
        
        for params_i, params in enumerate(grid):

            # ......................................................................    
            # GET PARAMS FOR ONE PARAM COMBINATION
            # ...................................................................... 

            # UPDATE MODEL ID, 
            model_ID +=1
            if track_progres==True:
                print('.', end="")
            else:
                pass

            # SET PARAMETERS
            pca_axes_nr        = params["pca"]
            #model_params_dct   = dict(zip(params["pc"],[params[x] for x in params["pc"]])) # not used here
            random_state_nr    = params["random_state_nr"]

            # store random nr to check if you need to re-load the data, 
            if model_ID==0:
                random_state_nr_inmemory = random_state_nr
            else:
                pass       
    
    
            # ......................................................................    
            # LOAD DATA
            # ......................................................................  
            '''
                Conditionally, - only if something has chnaged or it is thw 1st run,
            '''
            
            if model_ID>0 and random_state_nr_inmemory==random_state_nr and samples_to_drop_detected==False:
                affix_to_info_on_loaded_datasubset = " - no loading, using copy of data from last run, conditions were unchanged"
                pass            
            
            
            elif model_ID==0 or random_state_nr_inmemory!=random_state_nr or samples_to_drop_detected==True:
                # update random nr, 
                random_state_nr_inmemory = random_state_nr       
                affix_to_info_on_loaded_datasubset = ""  # just a message to knwo if the data were loaded again, 
     

                ####### LOSD TRAIN; VALID AND TEST SUBSETS #########################################################
                ''' 
                    in case, validation subset is created from train subset, 
                    it will be ommitted and loaded in the next step (3)
                '''
                
                # Step 1. Create dictionaries to store data 
                xy_data_dct = dict()
                xy_labels_dct = dict()
                xy_idx_dct = dict()            
            
            
                # Step 2. Add data bact/labels to sx_dict
                for xy_i, xy_name in enumerate(xy_names):
                    # get df-subset with filenames and paths
                    r_filter           = batchfile_table.loc[:, file_namepath_table_dict["subset_name"]]==xy_name
                    batchfiles_to_load = pd.DataFrame(batchfile_table.loc[r_filter, :])

                    # ommit valid datasets it not provided separately 
                    '''
                        they may not be available and must be created from train set
                    '''
                    # .... wait if there is not subset designated for valid, 
                    if xy_i==1 and valid_subset_name==None:
                        pass
                
                    # .... proceed using separate source data (train and valid were already separated)
                    else:
                        
                        # - a - load individual batches that will create one subset
                        '''
                                load and join data and batch label tables 
                                from all batches for a given dataset
                        '''
                        for row_i, row_nr in enumerate(list(range(batchfiles_to_load.shape[0]))):
                            
                            # - a.1 - find filenames and paths in the table
                            one_data_batch_filename  = batchfiles_to_load.loc[:, file_namepath_table_dict["batch_data_file_name"]].iloc[row_nr] 
                            one_data_batch_path      = batchfiles_to_load.loc[:, file_namepath_table_dict["batch_data_file_path"]].iloc[row_nr] 
                            # ...
                            one_batch_label_filename = batchfiles_to_load.loc[:, file_namepath_table_dict["batch_labels_file_name"]].iloc[row_nr] 
                            one_batch_label_path     = batchfiles_to_load.loc[:, file_namepath_table_dict["batch_labels_file_path"]].iloc[row_nr] 
             
                            # - a.2 - load, and concatenate
                            '''
                                check if paths were not enforced in funciton parameters, 
                            '''
                            if row_i==0:
                                if PATH_batch_data==None: 
                                    os.chdir(one_data_batch_path)
                                else: 
                                    os.chdir(PATH_batch_data)            
                                encoded_img_batch = np.load(one_data_batch_filename)
                                
                                # ......
                                if PATH_batch_labels==None: 
                                    os.chdir(one_batch_label_path)
                                else: 
                                    os.chdir(PATH_batch_labels)     
                                batch_labels = pd.read_csv(one_batch_label_filename)
                                batch_labels.reset_index(drop=True, inplace=True) # to be sure :)
                            
                            else:    
                                if PATH_batch_data==None: 
                                    os.chdir(one_data_batch_path)
                                else: 
                                    os.chdir(PATH_batch_data) 
                                encoded_img_batch = np.r_[encoded_img_batch, np.load(one_data_batch_filename)]
                                
                                # ......
                                if PATH_batch_labels==None: 
                                    os.chdir(one_batch_label_path)
                                else: 
                                    os.chdir(PATH_batch_labels)  
                                batch_labels = pd.concat([batch_labels, pd.read_csv(one_batch_label_filename)], axis=0)
                                batch_labels.reset_index(drop=True, inplace=True)
                                
     
                        # - b - Add loaded data to dict
                        if unit_test==False:
                            xy_data_dct[xy_name]   = encoded_img_batch
                            xy_labels_dct[xy_name] = batch_labels
                            xy_idx_dct[xy_name]    = np.arange(batch_labels.shape[0], dtype="int")    
                        
                        else:
                            '''
                                assign names to dict which using unit test=True 
                                 it is because xy_names in case of unit test, are [train, train, train]
                                 ie only one set would be loaded and saved, and no transformations later on possible
                            '''
                            unit_set_xy_names = ["train", "valid", "test"]
                            xy_data_dct[unit_set_xy_names[xy_i]]   = encoded_img_batch
                            xy_labels_dct[unit_set_xy_names[xy_i]] = batch_labels
                            xy_idx_dct[unit_set_xy_names[xy_i]]    = np.arange(batch_labels.shape[0], dtype="int")                        
                
                
                ####### CREATE VALID DATASETS FROM TRAIN SET #########################################################

                # Step 3. create valid dataset from train data if necessarly, 
                if unit_test==False:
                    if valid_subset_name==None and train_proportion!=None:
                        
                        # Split data into train/test sets
                        xy_data_dct[train_subset_name], xy_data_dct["valid"], xy_labels_dct[train_subset_name], xy_labels_dct["valid"] = train_test_split(
                            xy_data_dct[train_subset_name], xy_labels_dct[train_subset_name], 
                            train_size=train_proportion, 
                            test_size=(1-train_proportion),
                            random_state=random_state_nr 
                        )
                        
                        # get xy_idx to identify raw images in train/valid datasets, 
                        _, _, xy_idx_dct[train_subset_name], xy_idx_dct["valid"] = train_test_split(
                            xy_idx_dct[train_subset_name], np.arange(xy_idx_dct[train_subset_name  ].shape[0], dtype="int"), 
                            train_size=train_proportion, 
                            test_size=(1-train_proportion),
                            random_state=random_state_nr    # Caution, random_state_nr must be the same as in the above, 
                        ) 
                    else:
                        pass
                else:
                    pass
                
                ####### Correct subset names for unit test #########################################################

                # Step 4. Update xy_names
                if unit_test == False:
                    xy_names_loaded = xy_names.copy()
                else:
                    xy_names_loaded = ["train", "valid", "test"] # otherwise it is only train, train, train

                    
            # ......................................................................    
            # remove classes that shodul be dropped out (its not the mask - work only for entire classes)
            # ...................................................................... 
            if samples_to_drop_detected==True:     
                info_on_nr_of_dropped_items = dict()
                for ii, xy_name in enumerate(xy_names_loaded):     
                    # find indexes of samples that can be used 
                    idx_without_dropped_samples = np.where(xy_labels_dct[xy_name].loc[:,colname_with_classname_in_batch_labels_table]!=dropout_value)[0]
                    
                    # update each array/df 
                    '''
                        data and idx == array, labels==pd.df) 
                    '''
                    dataarr_before_the_drop = xy_data_dct[xy_name].shape[0]
                    # ...
                    xy_data_dct[xy_name]   = xy_data_dct[xy_name][idx_without_dropped_samples]
                    xy_idx_dct[xy_name]    = xy_idx_dct[xy_name][idx_without_dropped_samples]
                    xy_labels_dct[xy_name] = pd.DataFrame(xy_labels_dct[xy_name].iloc[idx_without_dropped_samples,:])
                    # ...
                    info_on_nr_of_dropped_items[xy_name]=f"dropped {dataarr_before_the_drop-xy_data_dct[xy_name].shape[0]} items"   
            else:
                
                pass
   
            # ......................................................................    
            # INFO AFTER LOADING THE LOAD DATA
            # ......................................................................                 
                
            # info
            if verbose==True:
                print(f"\n /-/ {params_i} /-/ params combination:")
                print(f"-      {params}")
            else:
                pass
            
            # parht of info that I wish to see even when just tracking the progres
            if verbose==True or (track_progres==True and model_ID==0):   
                print(f"- DATA SUBSETS LOADED: {affix_to_info_on_loaded_datasubset}")
                for ii, xy_name in enumerate(xy_names_loaded):
                    if samples_to_drop_detected==True:
                        message_on_dropped_samples = f' -//- {info_on_nr_of_dropped_items[xy_name]}'
                    else:
                        message_on_dropped_samples = ""
                    # .....
                    if unit_test==True:
                        if ii==0:
                            print(f"    . {xy_name}: {xy_data_dct[xy_name].shape}{message_on_dropped_samples}")
                        else:
                            print(f"    . {xy_name}: {xy_data_dct[xy_name].shape} - unit test - copy of train set{message_on_dropped_samples}") 
                    else:
                        if ii==1 and train_proportion!=None:
                            print(f"    . {xy_name}: {xy_data_dct[xy_name].shape} - CREATED FROM TRAIN SET ({np.round(1-train_proportion, 3)} of train set){message_on_dropped_samples}")
                        else:
                            print(f"    . {xy_name}: {xy_data_dct[xy_name].shape}{message_on_dropped_samples}")
            else:
                pass
            
            
            # ......................................................................    
            #  DATA PREPROCESSING
            # ......................................................................            
            
            # copy to make tranformation without reloading data
            xy_data_dct_final = xy_data_dct.copy()
            
            # correction   
            'set train dataset name'
            if unit_test==True:
                train_subset_name_used_in_xy_names = "train"
            else:
                train_subset_name_used_in_xy_names = train_subset_name   
                               
            ##### STEP 1. PCA,.................................................... 
            if pca_axes_nr!=0:
                # Train PCA model and use it to tranfomr data
                pca = PCA(n_components=pca_axes_nr) # it will use max nr of components == nr of features in dataset !
                pca.fit(xy_data_dct_final[train_subset_name], y=None) # Unsupervised learning, no y variable
                # ...
                for xy_name in xy_names_loaded:
                        xy_data_dct_final[xy_name] = pca.transform(xy_data_dct_final[xy_name])
            else:
                pass
                        
            ##### STEP 2. encode batch_labels,...................................................            
            xy_labels_dct_encoded = dict()
            for xy_name in xy_names_loaded:
                xy_labels_dct_encoded[xy_name] = xy_labels_dct[xy_name].classname.map(class_encoding)

                
            # ......................................................................    
            #  BASELINE
            # ......................................................................            

            # Create Most frequet baseline, 
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(xy_data_dct_final[train_subset_name_used_in_xy_names].astype(np.float), xy_labels_dct_encoded[train_subset_name_used_in_xy_names].astype(int))
            # ..
            baseline_acc = dict()
            for xy_name in xy_names_loaded:
                baseline_acc[f"baseline_acc_{xy_name}"] = dummy.score(xy_data_dct_final[xy_name], xy_labels_dct_encoded[xy_name])
            if verbose==True:
                print("- RESULTS:")
                print(" - ", model_ID, baseline_acc)    
            else:
                pass                   


            # ......................................................................    
            #  KERAS MODELS !
            # ......................................................................            
            'these are selected models, in the future I woudl like make this section much more advanced'

            ###### STEP 1. select the model ..............................................
            
            # from keras import backend as K
            K.clear_session()    

            # create model
            if params["model"]=="one_layer":
                model = create_keras_one_layer_dense_model(
                    input_size = xy_data_dct_final[train_subset_name_used_in_xy_names].shape[1],
                    output_size = len(list(class_encoding.keys())),
                    verbose = verbose,
                    **params
                    )            

            if params["model"]=="two_layers":
                model = create_keras_two_layer_dense_model(
                    input_size = xy_data_dct_final[train_subset_name_used_in_xy_names].shape[1],
                    output_size = len(list(class_encoding.keys())),
                    verbose = verbose,
                    **params
                    )

            # define early stopping - End training when acc stops improving (optional)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=params["EarlyStopping__patience"], 
                restore_best_weights=True
            )
              
            # caulate the weight for each class to deal with class imbalance
            X_tr__class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(xy_labels_dct_encoded[train_subset_name_used_in_xy_names].astype(int)),
                                                 xy_labels_dct_encoded[train_subset_name_used_in_xy_names].astype(int))
            
            # Fit model
            history = model.fit(
                x               =xy_data_dct_final[train_subset_name_used_in_xy_names].astype(np.float), # samples are subdivided internally, 
                y               =xy_labels_dct_encoded[train_subset_name_used_in_xy_names].astype(int),
                validation_data =(xy_data_dct_final["valid"], xy_labels_dct_encoded["valid"]),
                #validation_split=params['fit__validation_split'], 
                batch_size       =params['fit__batch_size'], 
                epochs           =params["fit__epoch"],
                shuffle          =True, # Shuffle training samples
                callbacks        =[early_stopping],
                class_weight     =X_tr__class_weights,
                verbose          =model_fit__verbose# no info, 
            )

            
            ##### STEP 2. get accuracy results .............................................

            # ... to store results, 
            model_acc = dict()
            loss_acc  = dict()
            n         = 3              # use last 3 results in history, 
            
            # ... train & valid
            acc_results = pd.DataFrame(history.history).iloc[-n::,:].mean(axis=0)
            model_acc["model_acc_train"]   = acc_results.loc["acc"]  
            model_acc["model_acc_valid"]   = acc_results.loc["val_acc"]
            model_acc["model_loss_train"]  = acc_results.loc["loss"]
            model_acc["model_loss_valid"]  = acc_results.loc["val_loss"]     
            
            # ... test datasets, 
            if len(xy_names_loaded)>2:
                for xy_name in xy_names_loaded[2::]:
                    loss, acc = model.evaluate(xy_data_dct_final[xy_name], xy_labels_dct_encoded[xy_name], verbose=0)
                    model_acc[f"model_acc_{xy_name}"]   = acc
                    model_acc[f"model_loss_{xy_name}"]  = loss 
            else:
                pass
 
            
            #### STEP 3. fixing small legacy issue - keeping all results the same across entire project
            knn_kneighbors = dict()
            for xy_name in xy_names_loaded:
                knn_kneighbors[xy_name] = None              

                
            # ......................................................................    
            #  COLLECT THE RESULTS
            # ......................................................................              
            '''
                acc_restuls_and_params were added to all objects in case 
                I woudl have some dounbts about results origine
            '''
            # - 1 - collect acc_restuls_and_params
            acc_restuls_and_params = {
                     "model_ID": model_ID,
                     "run_name": run_name,
                     "method": method,
                     "dataset_name": dataset_name, 
                     "dataset_variant": dataset_variant,
                     "module": module_name,
                     "unit_test":unit_test,
                     # ....
                     **baseline_acc,
                     **model_acc,
                     **loss_acc,       # nn only 
                     **params,
                     "pca_components_used":pca_axes_nr # legacy, 
            }
            model_acc_and_parameters_list.append(acc_restuls_and_params) # in list, so it can be used as pd.df immediately, 

            # - 2 - save model history, 
            model_history_dict[model_ID] = {
                "model_history": pd.DataFrame(history.history),
                "acc_restuls_and_params":  acc_restuls_and_params
            }
        
            # - 3 - store params, 
            "to allow future runs"
            model_parameters_dict[model_ID] = {
              "acc_restuls_and_params":  acc_restuls_and_params,  # when added, I can easily control what I am selecting in other 
              "params":params
              }            
            
            # - 4 - Collect Model predictions, 
            if store_predictions==True:
                one_model_predictions = dict()
                for xy_name in xy_names_loaded:
                    # make predictions and decode them,
                    predictions             = model.predict_classes(xy_data_dct_final[xy_name])
                    decoded_predictions     = pd.Series(predictions).map(class_decoding).values
                    model_predictions_proba = model.predict_proba(xy_data_dct_final[xy_name])
                    decoded_y_labels        = pd.Series(xy_labels_dct_encoded[xy_name]).map(class_decoding).values
                    # ...
                    one_model_predictions[xy_name] = {
                            "idx_in_batch":            xy_idx_dct[xy_name],
                            "original_labels":         decoded_y_labels, 
                            "model_predictions":       decoded_predictions, 
                            "model_predictions_proba": model_predictions_proba,
                            "acc_restuls_and_params":  acc_restuls_and_params,
                            "class_encoding":          class_encoding,
                            "class_decoding":          class_decoding,
                            "batch_labels_sf":         xy_labels_dct[xy_name],
                            "knn_kneighbors":          knn_kneighbors[xy_name]
                    }# added, in case I woudl have some dounbts about results origine, 

                    # and finally, add this to the big dict wiht all the results, 
                    model_predictions_dict[model_ID] = one_model_predictions

            else:
                model_predictions_dict[model_ID] = None
        
            # PLOT THE RESULTS ,......................    

            if plot_history==True:
              
                print(f"\n - - - {params_i} - - - ")
                print(acc_restuls_and_params)
              
                plot_NN_loss_acc(
                    model_history_df=pd.DataFrame(history.history), 
                    title=f"model history - {dataset_variant}", 
                    n_mean=3, 
                    figsize=(8,4), 
                    top=0.75
                )
            else:
                pass
            
        if track_progres==True:
            print(f"\nDONE - {pd.to_datetime('now')}",end="\n\n")
        else:
            pass
        
        # ..................................................
        return model_acc_and_parameters_list, model_predictions_dict, model_parameters_dict, model_history_dict        
        
        
        