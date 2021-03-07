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
from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch, load_raw_img_batch_with_custom_datagen
from src.utils.tools_for_plots import create_class_colors_dict
from src.utils.example_plots_after_clustering import plot_img_examples, create_spaces_between_img_clusters, plot_img_examples_from_dendrogram
from src.utils.annotated_pie_charts import annotated_pie_chart_with_class_and_group, prepare_img_classname_and_groupname


#############################################
import tensorflow as tf
import PIL.Image as Image
import tensorflow.keras as keras
from tensorflow.keras import backend as K # used for housekeeping of tf models, 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
def create_convNN(*, input_size, output_size, kwargs, verbose=False):
    ''' function to build cnn, with 2 convolutions, one hidden layer and one  
        note: its not mistake with kwags, - i kept it intentionally as packed distionary, 
              to allow to read parameter sourse in the code'
    '''
    run=True
    K.clear_session() 
    
    if run==True:
        # Convolutional Network, ........................
        model = keras.Sequential()

        #.. 1st cnn, layer
        model.add(keras.layers.Conv2D(
            filters=kwargs['Conv2D_1__filters'], 
            kernel_size=kwargs['Conv2D_1__kernel_size'], 
            strides=kwargs['Conv2D_1__stride'],
            activation=kwargs['Conv2D_1__activation'], 
            input_shape=input_size
            ))

        #.. maxpool 1.
        model.add(keras.layers.MaxPool2D(pool_size=kwargs['MaxPool2D_1__pool_size']))

        #.. 2nd cnn layer, 
        model.add(keras.layers.Conv2D(
            filters=kwargs['Conv2D_2__filters'], 
            kernel_size=kwargs['Conv2D_2__kernel_size'], 
            strides=kwargs['Conv2D_2__stride'],
            activation=kwargs['Conv2D_2__activation'], 
            ))

        #.. maxpool 2, 
        model.add(keras.layers.MaxPool2D(pool_size=kwargs['MaxPool2D_2__pool_size']))


        # flatten the results, .........................
        model.add(keras.layers.Flatten())
        
        
        # dense nn, ....................................            
        
        if kwargs["model"]=="two_dense_layers":
        
            #.. First hidden layer
            model.add(Dense(
                units=kwargs['h1_unit_size'], 
                activation=kwargs["h1_activation"],
                kernel_regularizer=tf.keras.regularizers.l2(kwargs['h1_l2']),
                kernel_initializer=initializers.VarianceScaling(scale=2.0, seed=0)
            ))
            model.add(tf.keras.layers.Dropout(kwargs["h1_Dropout"]))

        else:
            pass
        
        #.. Output layer
        model.add(Dense( 
            units=output_size, 
            activation=kwargs["out_activation"],
            kernel_regularizer=tf.keras.regularizers.l2(kwargs['out_l2']),
            kernel_initializer=initializers.VarianceScaling(scale=1.0, seed=0)
        ))  

        # Define Loss Function and Trianing Operation 
        model.compile(
            optimizer= kwargs["optimizer"],
            loss= losses.categorical_crossentropy,
            metrics= kwargs["metrics"] # even one arg must be in the list
        )

        
        if verbose==True:
            model.summary()
        else:
            pass
        
        return model  


    
    
# Function, .....................................................................................                   
def cnn_gridSearch(*,                  
    grid, 
    method_name="cnn",
    module_name,
    PATH_raw,
    train_subset_name, # one names, str, !
    test_subset_names,
    # ...                    
    store_predictions=True,
    track_progres=True,
    verbose=False,
    model_fit_verbose=0,
    plot_history=False # applied only if verbose==True, 
):
             

    if track_progres==True:
        print(f"{module_name} _________________________________________ {pd.to_datetime('now')}")
    else:
        pass
    
    # dist to store results, 
    model_acc_and_parameters_list = list()
    model_predictions_dict = dict()
    model_history_dict = dict()
    
    # grid search, with cnn
    model_ID = -1
    for params in grid:

        model_ID +=1
        if track_progres==True:
            print('.', end="")
        else:
            pass        
        
        
        # CREATE DATA GENERATORS AND LOAD TEST DATA ...............................
        """ 
            but only for train and validation sets, 
            test set is crwate ad in all other functions that I did
        """
        
        # step 1. define generators, 
        
        # .. Image generator for train dasta, with data augmentaiton, 
        train_datagen = ImageDataGenerator(
            rescale=1/255, 
            validation_split=params["train_generator__validation_split"],
            horizontal_flip=True, 
            rotation_range=5, 
            #**params["train_generator__kwargs"]
        )
        
        # .. Image generator for train dasta, with data augmentaiton, 
        valid_datagen = ImageDataGenerator(
            rescale=1/255
        )
              
            
        # Step 2. Create Iterators,  - returns one-hot encoded labels!
        
        # .. for train dataset
        trainset_iter = train_datagen.flow_from_directory(
            os.path.join(PATH_raw, train_subset_name), 
            batch_size=params["train_generator__batch_size"], 
            target_size=params["generator__target_size"],
            shuffle=True, # must be shuffle=True, !
            subset='training' # function name
        )

        # .. for validation data, no shuffle, made from the same dataset as train data, 
        validset_iter = train_datagen.flow_from_directory(
            os.path.join(PATH_raw, train_subset_name), 
            batch_size=params["train_generator__batch_size"], 
            target_size=params["generator__target_size"],
            shuffle=False,
            subset='validation'# function name
        )
            
        # .. define encoded classes, - very important !
        class_encoding = trainset_iter.class_indices
        class_decoding = dict(zip(list(list(class_encoding.values())), list(class_encoding.keys()))) # reverse on class_encoding,
            
        # Step 3. Load test data - done once, for all datasets, 
        "with modified generators that also collect data on img_labels and idx in the file"     
        raw_img_batch__te, batch_labels_df__te = load_raw_img_batch_with_custom_datagen(                                                
            path=PATH_raw, 
            subset_names=test_subset_names,  
            n_next_datagen = 1, 

            # --- ImageDataGenerator_kwargs
            ImageDataGenerator_kwargs = params["train_generator__kwargs"],

            # --- for datagen.flow_from_directory()
            datagen__target_size=params["generator__target_size"],   

            # --- generator stuff, solved with my script, 
            subset_batch_size = None, # None == all samples in the file will be loaded, once,                                           
            shuffle_batch = False,  # 
            shuffle_all = False,                     
            # --- function, 
            verbose=verbose
        )      
        
        # .. adapt for making predictions with the model, 
        raw_img_batch__te = raw_img_batch__te.astype(np.float)
        labels_te = pd.Series(batch_labels_df__te.classname).map(class_encoding).values.astype("int")          

        
        
        
        
        ######################################################################   
            
        # CREATE AND TRAIN THE MODEL ,................
        "params dict is used here to provide imputs for parameter values"    
        
        # model,         
        K.clear_session()   # also done internally,   
        model=create_convNN(
            input_size=(params["generator__target_size"][0], params["generator__target_size"][1], 3), # RGB only !, 
            output_size=trainset_iter.num_classes, 
            kwargs=params, 
            verbose=False
        )
        
        # Early stopping - End training when acc stops improving (optional)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["EarlyStopping__patience"])

        # Train model
        history = model.fit_generator(
            generator=trainset_iter, # you provide iterators, instead of data, 
            validation_data=validset_iter, 
            epochs=params["fit_generator__epoch"], 
            callbacks=[early_stopping],   # functions that can be applied at diffeerent stages to fight overfitting, 
            verbose=model_fit_verbose,
        )  
        
        # PLOT THE RESULTS ,......................    
        if plot_history==True:
            plot_NN_loss_acc(
                model_history_df=pd.DataFrame(history.history), 
                title=f"model_ID: {model_ID}", 
                n_mean=params["EarlyStopping__patience"], 
            )
            model.summary()
            print(params)
                
        else:        
            pass

            
            
        ###################################################################### 
                
        # EVALUTE MODEL ACC, .......................... 
        model_acc = dict()
        loss_acc = dict()

        # ... train and valid data, 
        n = 3 # early stopping steps taken into account, 
        acc_results = pd.DataFrame(history.history).iloc[-n::,:].mean(axis=0)
        model_acc["model_acc_train"] = acc_results.loc["acc"]  
        model_acc["model_acc_valid"] = acc_results.loc["val_acc"]
        model_acc["model_loss_train"] = acc_results.loc["loss"]
        model_acc["model_loss_valid"] = acc_results.loc["val_loss"]     
        
        # to make everything easy, the first 
        
        # .. spacers_ce expects binary matrices, with column for each class, 
        y_te_df = pd.Series(labels_te)
        y_binary = pd.get_dummies(y_te_df).values
        
        #y_binary = to_categorical(labels_te)

        # .. predict
        loss, acc = model.evaluate(raw_img_batch__te,  y_binary, verbose=0)
        model_acc["model_acc_test"] = acc
        model_acc["model_loss_test"] = loss  
    
    
        # COLLECT THE RESULTS AND PREDICTIONS ,..............................  
        'acc_restuls_and_params were added to all objects in case I woudl have some dounbts about results origine,'

        # 1. acc_restuls_and_params
        acc_restuls_and_params = {
                 "random_state_nr": params["random_state"], # for backcompatibility, 
                 "model_ID": model_ID,
                 "method": method_name,
                 "module": module_name,
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
            
            # .. make predictions and decode them,
            predictions               = model.predict_classes(raw_img_batch__te)
            decoded_predictions       = pd.Series(predictions).map(class_decoding).values
            model_predictions_proba   = model.predict_proba(raw_img_batch__te)
            decoded_y_labels          = pd.Series(labels_te).map(class_decoding).values
            
            # .. create structure used with all other models, 
            one_model_predictions = {
                "test": {
                        "idx_in_batch":            np.arange(batch_labels_df__te.shape[0]), 
                                                # its like that, because >I assume I will always use some datasets in combination
                        "original_labels":         decoded_y_labels, 
                        "model_predictions":       decoded_predictions, 
                        "model_predictions_proba": model_predictions_proba,
                        "acc_restuls_and_params":  acc_restuls_and_params,
                        "class_encoding":          class_encoding,
                        "class_decoding":          class_decoding,
                        "batch_labels_df":         batch_labels_df__te
                }}

            # .. and finally, add this to the big dict wiht all the results, 
            model_predictions_dict[model_ID] = one_model_predictions
            
        else:
            model_predictions_dict[model_ID] = None


    if track_progres==True:
        print(f"\nDONE _________________________________________ {pd.to_datetime('now')}",end="\n\n")
    else:
        pass
    
    # ..................................................
    return model_acc_and_parameters_list, model_predictions_dict, model_history_dict








