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
from IPython.display import display
from tensorflow.keras import backend as K # used for housekeeping of tf models,
import PIL.Image as Image
import tensorflow.keras as keras
from tensorflow.keras import backend as K # used for housekeeping of tf models, 
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
from sklearn.utils import class_weight

from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch, load_raw_img_batch_with_custom_datagen, cnn_data_loader
from src.utils.logreg_tools import my_logredCV, plot_examples_with_predictions_and_proba, plot_examples_with_predictions_and_proba_gamma
# plot_NN_loss_acc, cnn_model_for_unit_test, basic_cnn_model_with_two_layers, basic_cnn_model_with_two_layers




# Function, ................................................................................
def plot_NN_loss_acc(*, model_history_df, title="", n_mean=3, figsize=(8,4), top=0.75):
    ''' small function to plot loss and accuracy over epoch using data created with history.history() keras functions, 
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
        
      
      
      
      


  
  
  
  
# Function, .................................................................................
def cnn_model_for_unit_test(*, input_size, output_size, verbose=False):
    ''' function to build cnn, with 2 convolutions, flatten and output layer with unit == class nr,  
        all parameters , except for input and output size are predefined, and tested to be working, '
        the network can be used with any image classyfication problem, but best tit to start from dataset for unit/function test
        that contains minimal 5-6 classes, and only 1 distint image i each classs, and the same picture is in validation, train and datasset
        it is expected to overfit and acquire acc==1 with all datasets in 50-60 epoch, 
    '''
    K.clear_session() 
    
    # .................................................................
    # Step 1. Create sequential model, 

    # Convolutional Network
    model = keras.Sequential()

    #.. 1st cnn, layer
    model.add(keras.layers.Conv2D(
        filters=64, 
        kernel_size=5, 
        strides=2,
        activation='relu', 
        input_shape=input_size
        ))

    #.. maxpool 1.
    model.add(keras.layers.MaxPool2D(pool_size=2))

    #.. 2nd cnn layer, 
    model.add(keras.layers.Conv2D(
        filters=64, 
        kernel_size=3, 
        strides=1,
        activation='relu'
        ))

    #.. maxpool 2, 
    model.add(keras.layers.MaxPool2D(pool_size=2))

    #.. flat the results,
    model.add(keras.layers.Flatten())

    #.. dense nn, 
    model.add(keras.layers.Dense(
        units=output_size, 
        activation='softmax'
        ))

    # .................................................................
    # Step 2. Compile the model with the Adam optimizer, 
    model.compile(optimizer="SGD", loss='sparse_categorical_crossentropy', metrics=['acc'])


    # .................................................................
    # Step 3. Add Early stopping - End training when acc stops improving (optional)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    if verbose==True:
        model.summary()
    else:
        pass
    
    return model#, early_stopping




  
  

# Function, .................................................................................
def basic_cnn_model_with_two_layers(*, input_size, output_size, params=None, verbose=False):
    ''' function to build cnn, with 2 convolutions, flatten and output layer with unit == class nr,  
        all parameters , except for input and output size are predefined, and tested to be working, '
        the network can be used with any image classyfication problem, but best tit to start from dataset for unit/function test
        that contains minimal 5-6 classes, and only 1 distint image i each classs, and the same picture is in validation, train and datasset
        it is expected to overfit and acquire acc==1 with all datasets in 50-60 epoch, 
    '''
    
    if params==None:
        params = dict()
        params["optimizer"] = "Adam"
        params["dense_one__units"] = 60
    else:
        pass
      
    K.clear_session() 
    
    # .................................................................
    # Step 1. Create sequential model, 

    # Convolutional Network
    model = keras.Sequential()

    
    #.. 1st cnn, layer
    model.add(keras.layers.Conv2D(
        filters=64, 
        kernel_size=5, 
        strides=2,
        activation='relu', 
        input_shape=input_size
        ))
    model.add(tf.keras.layers.Dropout(0.5))

    #.. maxpool 1.
    model.add(keras.layers.MaxPool2D(pool_size=2))

    
    #.. 2nd cnn layer, 
    model.add(keras.layers.Conv2D(
        filters=128, 
        kernel_size=3, 
        strides=1,
        activation='relu'
        ))
    model.add(tf.keras.layers.Dropout(0.5))

    #.. maxpool 3, 
    model.add(keras.layers.MaxPool2D(pool_size=2))    
    
    
    #.. flat the results,
    model.add(keras.layers.Flatten())
    
    #.. First hidden layer
    model.add(Dense(
        units=params["dense_one__units"], 
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=0)
    ))
    model.add(tf.keras.layers.Dropout(0.5))
    
    #.. dense nn, 
    model.add(keras.layers.Dense(
        units=output_size, 
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=initializers.VarianceScaling(scale=1.0, seed=0)
        ))

    # .................................................................
    # Step 2. Compile the model with the Adam optimizer, 
    model.compile(optimizer=params["optimizer"], loss='sparse_categorical_crossentropy', metrics=['acc'])


    # .................................................................
    # Step 3. Add Early stopping - End training when acc stops improving (optional)
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # set externally, 
    
    if verbose==True:
        model.summary()
    else:
        pass
    
    return model#, early_stopping



  
  
  
  
  
  
# Function, .................................................................................  
def cnn_gridSearch_customDataLoader(    
    model,
    method_name,
    module_name,
    grid,
    dataset_name,
    class_colors,
    # ...
    train_subset_names,
    valid_subset_names,
    test_subset_names,      
    PATH_raw,
    # ...
    augment_valid_data = False,  
    store_predictions = True,
    # ...
    model_unit_test = False, # train dat are used as validation and test subsets, all corresponding params are ignored,
    # ...
    verbose=False,
    track_progres=False,
    model_fit__verbose=0,    
    plot_history=False, # applied only if verbose==True,
):
    
    
    # ............................................................................... #
    # dist to store results, 
    model_acc_and_parameters_list = list()
    model_predictions_dict = dict()
    model_history_dict = dict()  
    model_ID = -1
    params_i = 0
    # ............................................................................... #    
        

    # start,
    r=True
    if r==True:
        for params_i, params in enumerate(grid):
                
            # SET PARAMS, .............................................. 
            model_ID +=1
            img_size = params["img_size"]
            random_state_nr = params["random_state_nr"]
            data_loader_params = params["data_loader_params"]
            model_params = params["model_params"]
            
            # ... info, 
            if verbose==True:
                print(f"{module_name}//{random_state_nr}//{params_i+1}of{len(grid)} --- {pd.to_datetime('now')}")
            else:
                pass   
            # ... or a bit smaller info, 
            if track_progres==True:
                if params_i==0 : print(f"grid analysis")
                if params_i>0 and params_i<(len(grid)-1) and (params_i%10)==0:     print(f"({params_i})", end="")
                if params_i>0 and params_i<(len(grid)-1) and (params_i%10)!=0:     print(f".", end="")
                if params_i>0 and params_i==(len(grid)-1):                         print(f"final combination")
            else:
                pass 
                        
                
    
    
            # DATA PREPARATION, .......................................
            X_dct, y_dct, batch_labels_dct, idx_y_dct, class_encoding,  class_decoding = cnn_data_loader(
                PATH_raw           = PATH_raw, 
                train_subset_names = train_subset_names,
                valid_subset_names = valid_subset_names,  # if None, it will be created from train data, 
                test_subset_names  = test_subset_names,   # if None, it will not be created at all, 
                model_unit_test    = model_unit_test,     # if True, (train, test and valid datasets are created using Trains datasets without augmentation, in one batch)
                augment_valid_data = augment_valid_data,  # if True, valid data are processed in the same way as train data, 
                img_size           = img_size,            # can be used only if params are not given, otherwise it shodul be there, 
                params             = data_loader_params, # extracted at the beginning of the loop, step
                verbose            = verbose,
                random_state_nr    = random_state_nr
            )
            Xy_names = list(X_dct.keys())

            
            
            
            
            # CREATE AND TRAIN CNN MODEL ,.............................  
            model_imput_size = model_imput_size = (img_size[0], img_size[1], 3) # for RGB, we need to define 3 layers, 
            model_output_size = len(class_encoding.keys())
            
            # a) create the model,         
            K.clear_session()   # also done internally,   
            model = basic_cnn_model_with_two_layers(
                input_size         =model_imput_size, # RGB only !, ##### NEW FUNCTION, ......................... !!!!
                output_size        =model_output_size, 
                params             =model_params,
                verbose            =False,                             
                )                                                    

            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=model_params['EarlyStopping__patience'], 
                restore_best_weights=True
                )
            # b) caulate the weight for each class to deal with class imbalance
            X_tr__class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_dct["train"]),
                                                 y_dct["train"])
            # c) train the model
            history = model.fit(
                    x                      =X_dct["train"], 
                    y                      =y_dct["train"],
                    validation_data        =(X_dct["valid"], y_dct["valid"]), 
                    batch_size             =model_params["fit__batch_size"], 
                    epochs                 =model_params["fit__epochs"],
                    shuffle                =model_params["fit__shuffle"], # Shuffle training samples
                    callbacks              =[early_stopping],
                    class_weight           =X_tr__class_weights,
                    verbose                =model_fit__verbose # as param in main function, 
                )
   
            
            
        
        
            # EVALUATE THE MODEL ,............................. 
            n = model_params['EarlyStopping__patience']# early stopping steps taken into account, 
            acc_results = pd.DataFrame(history.history).iloc[-n::,:].mean(axis=0)
            # ...
            model_acc = dict()
            model_loss = dict()            
            for xy_name in Xy_names:
                loss, acc = model.evaluate(X_dct[xy_name], y_dct[xy_name], verbose=0)
                model_acc[f"model_acc_{xy_name}"] = acc
                model_loss[f"model_loss_{xy_name}"] = loss  

                
                
                
            # COLLECT THE RESULTS ,..............................  
            'acc_restuls_and_params were added to all objects in case I woudl have some dounbts about results origine,'

            # 1. acc_restuls_and_params
            acc_restuls_and_params = {
                     "random_state_nr": random_state_nr, # for backcompatibility, 
                     "model_ID": model_ID,
                     "method": method_name,
                     "module": module_name,
                     "img_size" : img_size,
                     #**baseline_acc,
                     **model_acc,
                     **model_loss,
                     **data_loader_params,
                     **model_params
            }
            model_acc_and_parameters_list.append(acc_restuls_and_params) # in list, so it can be used as pd.df immediately, 

            # 2. save model history, 
            model_history_dict[model_ID] = {
                "model_history": pd.DataFrame(history.history),
                "acc_restuls_and_params":  acc_restuls_and_params}

            # 3. Model predictions, 
            """collect all model predictions also for test and valid datasets 
               to have nice comparisons on errors and problematic files"""
            if store_predictions==True:
                one_model_predictions = dict()
                for xy_name in Xy_names:
                    # make predictions and decode them,
                    predictions               = model.predict_classes(X_dct[xy_name])
                    decoded_predictions       = pd.Series(predictions).map(class_decoding).values
                    model_predictions_proba   = model.predict_proba(X_dct[xy_name])
                    decoded_y_labels          = pd.Series(y_dct[xy_name]).map(class_decoding).values
                        # ...
                    one_model_predictions[xy_name] = {
                            "idx_in_batch":            idx_y_dct[xy_name],
                            "original_labels":         decoded_y_labels, 
                            "model_predictions":       decoded_predictions, 
                            "model_predictions_proba": model_predictions_proba,
                            "acc_restuls_and_params":  acc_restuls_and_params,
                            "class_encoding":          class_encoding,
                            "class_decoding":          class_decoding,
                            "batch_labels":             batch_labels_dct[xy_name]
                    }# added, in case I woudl have some dounbts about results origine, 

                # and finally, add this to the big dict wiht all the results, 
                model_predictions_dict[model_ID] = one_model_predictions

            else:
                model_predictions_dict[model_ID] = None

                
                
            # BASIC INFO .............................................................................. 
            if verbose==True or plot_history==True:
                print("..............................................................")
                print("MAIN RESULTS")  
                print(model_acc)  
                print(model_loss)  
                print("..............................................................")
            else:
                pass                
                                      

            # PLOTS ..................................................................................     
            if plot_history==True:
                if model_unit_test==True:
                    dtype_for_plot = "train"
                    load_datasets_for_plot = train_subset_names
                if model_unit_test==False:
                    dtype_for_plot = "test"
                    load_datasets_for_plot = test_subset_names
                    
                print(model.summary())
                print(model_params)  
                print(data_loader_params)
                    
                # .. plot loss and accuracy values history over epoch
                test_acc_results = model_predictions_dict[model_ID][dtype_for_plot]['acc_restuls_and_params']["model_acc_test"]        
                plot_NN_loss_acc(
                    title=f"Best Performing Model, Acc for test data = {str(np.round(test_acc_results,3))}",
                    model_history_df=model_history_dict[model_ID]['model_history'], 
                    n_mean=3,
                    figsize=(6,3),
                    top=0.75
                )

                # .. plot loss and accuracy values history over epoch
                plot_examples_with_predictions_and_proba_gamma( 
                    model_ID=model_ID,
                    model_predictions_dict= model_predictions_dict, 
                    module_name="my network", 
                    dataset_name=dataset_name,                              
                    subset_name=[dtype_for_plot], # denotes test predicitons made for test_subset_names
                    img_batch_subset_names=load_datasets_for_plot, 
                    path_to_raw_img_batch=PATH_raw,
                    class_colors=class_colors,
                    make_plot_with_img_examples=True, # use False, to have only pie charts with classyfication summary                                         
                    max_img_per_col=7,
                    plot_classyfication_summary=True
                )
            else:
                pass
            
    # ..................................................
    return model_acc_and_parameters_list, model_predictions_dict, model_history_dict

            