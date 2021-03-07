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

def create_keras_one_layer_dense_model(*,
    input_size, 
    output_size, 
    verbose=False,
    **kwargs
):

    """
        
        Notes:
        https://www.tensorflow.org/tutorials/keras/save_and_load
    """
    
    
    # ...................................................
    # Create model
    model = Sequential()

    #.. add fully connected layer
    model.add(Dense(
        input_dim=input_size,  # IE 784 PIXELS, !
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

    
    
    
    



# Fuction ...................................................................................

def create_keras_three_layer_dense_model(*,
    input_size, 
    output_size, 
    verbose=False,
    **kwargs
                                       
):

    # ...................................................
    # Create model
    "use the same parameters for 1st and seocnd layer"
    model = Sequential()
    
    #.. First hidden layer
    model.add(Dense(
        input_dim=input_size,
        units=kwargs['h1_unit_size'], 
        activation=kwargs["h1_activation"], 
        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=0)
    ))
    
    #.. Second hidden layer
    model.add(Dense(
        units=kwargs['h1_unit_size'], 
        activation=kwargs["h1_activation"], 
        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=0)
    ))    
    
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
        
        
    
    
    
    
    
  


# Function , .........................................................

def denseNN_grid_search(*, 
    dataset_name,
    method_name,
    module_name,
    PATH_encoded,
    train_subset_names,
    test_subset_names,
    # ...                    
    class_encoding,
    grid,
    store_predictions=True,
    track_progres=True,
    verbose=False,
    plot_history=False # applied only if verbose==True, 
):
    
    # dist to store results, 
    model_acc_and_parameters_list = list()
    model_predictions_dict = dict()
    model_history_dict = dict()
    class_decoding = dict(zip(list(list(class_encoding.values())), list(class_encoding.keys()))) # reverse on class_encoding, 
    
    # .. 
    if track_progres==True:
        print(f"{module_name} _________________________________________ {pd.to_datetime('now')}")
    else:
        pass

    # Grid search, 
    model_ID = -1 # id number for each model, its predictions, I started with -1 so the first id will be 0 !
    for params in grid:    

        # PARAMETERS, ...................................
        model_ID +=1
        Xy_names = ["train", "valid", "test"] # these are internal names for datasets create with that function,
         # not the names of datatsets important that can ghave the same names, or other names, 
            
        if track_progres==True:
            print('.', end="")
        else:
            pass
        
        
        
        # LOAD & PREPARE THE DATA ,......................
        
        # find any logfile created while saving img files, 
        os.chdir(PATH_encoded)
        logfiles = []
        for file in glob.glob(f"{''.join([module_name,'_',dataset_name])}*_logfile.csv"):
            logfiles.append(file)
                
        # Load train data, 
        X_tot, batch_labels = load_encoded_imgbatch_using_logfile(logfile_name=logfiles[0], load_datasetnames=train_subset_names)
        X_tot = X_tot.astype(np.float)
        y_tot = pd.Series(batch_labels.classname).map(class_encoding).values.astype("int")
        
        # Load test data, 
        X_te, batch_labels = load_encoded_imgbatch_using_logfile(logfile_name=logfiles[0], load_datasetnames=test_subset_names)
        X_te = X_te.astype(np.float)
        y_te = pd.Series(batch_labels.classname).map(class_encoding).values.astype("int")
        idx_y_te = np.arange(y_te.shape[0]) # kep for compatibility issues

        # ... Split data into train/validation sets
        """ here it is done to prepare the script for future applications"""
        X_tr, X_valid, y_tr, y_valid = train_test_split(
                X_tot, y_tot, 
                train_size=params["train_test_split__train_size"], 
                test_size=(1-params["train_test_split__train_size"]),
                random_state=params["random_state"]
        )     
        
         # ... get xy_idx to identify raw images in train/valid datasets, 
        _, _, idx_y_tr, idx_y_valid = train_test_split(
                X_tot, np.arange(X_tot.shape[0], dtype="int"), 
                train_size=params["train_test_split__train_size"], 
                test_size=(1-params["train_test_split__train_size"]),
                random_state=params["random_state"]
        )

        # place all in dict,
        X_dct = dict(zip(Xy_names, [X_tr, X_valid, X_te]))
        y_dct = dict(zip(Xy_names, [y_tr, y_valid, y_te]))
        idx_y_dct = dict(zip(Xy_names, [idx_y_tr, idx_y_valid, idx_y_te]))            

        # SHUFFLE , ................................... 
        'only in case X_tot is used for NN training'
            
        # shuffle the samples in tot - otherwise the model will load batches, smaller then class, ie, one batch will often haven samples from only one class !
        # ... it will very fast went into overfitting with low accurqcy and huge loss for validation set, 
        idx = np.arange(X_tot.shape[0])
        my_seed = np.random.RandomState(params["random_state"])
        idx_mix = my_seed.choice(a=idx, size=idx.shape[0], replace=False)
        X_tot = X_tot[idx_mix,:].copy()
        y_tot = y_tot[idx_mix].copy()        
        

        # INFO , ................................... 
        if verbose==True:
            print(f"\n{''.join(['-']*40)}"); print(f"{''.join(['-']*40)}");print(f"{''.join(['-']*40)}")
            print(f'{model_ID}: {module_name}, logfie: {logfiles[0]}'); print(f"{''.join(['-']*40)}")
            print("PARAMETERS:"); print(f'{model_ID}: {params}')
            print("INPUT DATA DIMENSIONS:");
            for xyname in Xy_names:
                print(f"{xyname}: {X_dct[xyname].shape}")
        else:
            pass

        
        # BASELINE, ...............................
        'Create Most frequet baseline - done mainly for bakccompatibility'
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_dct["train"].astype(np.float), y_dct["train"].astype(int))
        # ..
        baseline_acc = dict()
        for xyname in Xy_names:
            baseline_acc[f"baseline_acc_{xyname}"] = dummy.score(X_dct[xyname], y_dct[xyname])

        if verbose==True:
            print(" --- ", model_ID, baseline_acc)    
        else:
            pass
            

        # CREATE AND TRAIN THE MODEL ,................
        "params dict is used here to provide imputs for parameter values"    
               
        # from keras import backend as K
        K.clear_session()    
            
        # create model
        if params["model"]=="one_layer":
            model = create_keras_one_layer_dense_model(
                input_size = X_tot.shape[1],
                output_size = len(list(class_encoding.keys())),
                verbose = verbose,
                **params
                )            
                  
        if params["model"]=="two_layers":
            model = create_keras_two_layer_dense_model(
                input_size = X_tot.shape[1],
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
        
        # Fit model
        history = model.fit(
            x=X_tot, # samples are subdivided internally, 
            y=y_tot,
            validation_split=params['fit__validation_split'], 
            batch_size=params['fit__batch_size'], 
            epochs=params["fit__epoch"],
            shuffle=True, # Shuffle training samples
            callbacks=[early_stopping],
            verbose=0# no info, 
        )
        
        

        # EVALUTE MODEL ACC, .......................... 
        model_acc = dict()
        loss_acc = dict()
        # ...
        n = params["EarlyStopping__patience"]# early stopping steps taken into account, 
        acc_results = pd.DataFrame(history.history).iloc[-n::,:].mean(axis=0)
        model_acc["model_acc_train"] = acc_results.loc["acc"]  
        model_acc["model_acc_valid"] = acc_results.loc["val_acc"]
        model_acc["model_loss_train"] = acc_results.loc["loss"]
        model_acc["model_loss_valid"] = acc_results.loc["val_loss"]     
        # ...
        loss, acc = model.evaluate(X_dct["test"], y_dct["test"], verbose=0)
        model_acc["model_acc_test"] = acc
        model_acc["model_loss_test"] = loss  
    
        # COLLECT THE RESULTS ,..............................  
        'acc_restuls_and_params were added to all objects in case I woudl have some dounbts about results origine,'

        # 1. acc_restuls_and_params
        acc_restuls_and_params = {
                 "random_state_nr": params["random_state"], # for backcompatibility, 
                 "model_ID": model_ID,
                 "method": method_name,
                 "module": module_name,
                 **baseline_acc,
                 **model_acc,
                 **params
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
            for xyname in Xy_names:
                # make predictions and decode them,
                predictions               = model.predict_classes(X_dct[xyname])
                decoded_predictions       = pd.Series(predictions).map(class_decoding).values
                model_predictions_proba   = model.predict_proba(X_dct[xyname])
                decoded_y_labels          = pd.Series(y_dct[xyname]).map(class_decoding).values
                    # ...
                one_model_predictions[xyname] = {
                        "idx_in_batch":            idx_y_dct[xyname],
                        "original_labels":         decoded_y_labels, 
                        "model_predictions":       decoded_predictions, 
                        "model_predictions_proba": model_predictions_proba,
                        "acc_restuls_and_params":  acc_restuls_and_params,
                        "class_decoding":          class_decoding
                }# added, in case I woudl have some dounbts about results origine, 

            # and finally, add this to the big dict wiht all the results, 
            model_predictions_dict[model_ID] = one_model_predictions
            
        else:
            model_predictions_dict[model_ID] = None

            
            
        # PLOT THE RESULTS ,......................    
                    
        if verbose==True and plot_history==True:
        
            #.. figure, axes, 
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
            fig.suptitle(f"{params}")

            #.. Plot accuracy values
            ax1.plot(history.history['loss'], label='train loss')
            ax1.plot(history.history['val_loss'], label='val loss')
            ax1.set_title('Validation loss {:.3f} (mean last 3)'.format(
                np.mean(history.history['val_loss'][-3:]) # last three values
            ))
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('loss value')
            ax1.grid(ls="--", color="grey")
            ax1.legend()

            #.. Plot accuracy values
            ax2.plot(history.history['acc'], label='train acc')
            ax2.plot(history.history['val_acc'], label='val acc')
            ax2.set_title('Validation accuracy {:.3f} (mean last 3)'.format(
                np.mean(history.history['val_acc'][-3:]) # last three values
            ))
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('accuracy')
            ax2.set_ylim(0,1)
            ax2.grid(ls="--", color="grey")
            ax2.legend()
            plt.show()
        
        else:
            pass
  

    if track_progres==True:
        print(f"\nDONE _________________________________________ {pd.to_datetime('now')}",end="\n\n")
    else:
        pass

    # ..................................................
    return model_acc_and_parameters_list, model_predictions_dict, model_history_dict
