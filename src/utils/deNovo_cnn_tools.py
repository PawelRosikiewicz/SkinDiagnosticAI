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
from sklearn.metrics import accuracy_score
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





    
    
    
    
# Function, ........................................................................    
def deNovoCNN_gridsearch(*,        
        # ... model/run description 
        method,
        grid,
        run_name,
        dataset_name,
        dataset_variant,
        module_name,
        # ... filenames & paths
        dataset_path,
        subset_names_to_load,
        xy_names, 
        # ... results and info, 
        store_predictions=True,
        track_progres=True,  
        verbose=True,
        plot_history=True,
        model_fit_verbose=1
    ):


    # ......................................................................    
    # Set up, 
    # ......................................................................    

    # - variables,
    model_ID = -1   # to start usnique counts of the models from 0 
    colname_with_classname_in_batch_labels_table = "classname"
    
    # - to store the results, 
    model_acc_and_parameters_list = list()
    model_predictions_dict        = dict()
    model_parameters_dict         = dict() # for iterations from grid
    model_history_dict            = dict()
    

    
    # ......................................................................    
    # GET PARAMS FOR ONE PARAM COMBINATION
    # ......................................................................         
    for params_i, params in enumerate(grid):
        model_ID        +=1

        
        #### -------------------------------------------------
        #### STEP 1. IMAGE GENERATORS  
        
        # ... train and valid subsets, 
        train_datagen = ImageDataGenerator(
            rescale          =1/255,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
            rotation_range=30,
            brightness_range=[0.2,1.8], 
            zoom_range=[0.75,1.25], # ±25%
            channel_shift_range=.1,
            #validation_split=params["validation_split"]
        )
        # ... valid subsets
        valid_datagen = ImageDataGenerator(rescale=1/255)        
  


        #### -------------------------------------------------
        #### STEP 2. ITERATORS FOR TRAIN AND VALID SUBSETS,  
        
        # ... for train dataset
        trainset = train_datagen.flow_from_directory(
            os.path.join(dataset_path, subset_names_to_load[0]), 
            batch_size     =params["batch_size"], 
            target_size    =params["img_size"],
            shuffle        =True
            #subset         ="train"
        )

        # for validation data, no shuffle, made from the same dataset as train data, 
        validset = valid_datagen.flow_from_directory(
            os.path.join(dataset_path, subset_names_to_load[1]), 
            batch_size     =params["batch_size"], 
            target_size    =params["img_size"],
            shuffle        =False,
            #subset        ="valid"
        )
      
        # collect class encoding/decoding - its not the standard one, 
        class_encoding  = trainset.class_indices
        class_decoding  = dict(zip(list(class_encoding.values()),list(class_encoding.keys())))

        

        
        #### -------------------------------------------------
        #### STEP 3. SEQUENCIAL MODEL, 
        
        input_size  = (params["img_size"][0], params["img_size"][1], 3)
        output_size = trainset.num_classes

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

        #.. hidden layer
        model.add(Dense(
            units=params["f1_units"], 
            activation="relu",
            #kernel_regularizer=tf.keras.regularizers.l2(0.001),
            kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=0)
        ))
        model.add(tf.keras.layers.Dropout(params["f1_dropout"]))

        #.. output nn, 
        model.add(keras.layers.Dense(
            units=output_size, 
            activation='softmax'
            ))

        # compile, 
        model.compile(optimizer=params["optimizer"], loss='categorical_crossentropy', metrics=['acc'])
        model.summary()


        
        #### -------------------------------------------------
        #### STEP 4. TRAIN MODEL, 
        
        # Train model
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["early_strop"])
        history = model.fit_generator(
            generator        =trainset,          # you provide iterators, instead of data, 
            validation_data  =validset, 
            epochs           =params["epoch"], 
            callbacks        =[early_stopping],   # functions that can be applied at diffeerent stages to fight overfitting, 
            verbose          =model_fit_verbose
        )
        
        # .. plot
        plot_NN_loss_acc(
                model_history_df=pd.DataFrame(history.history), 
                title=f"model_ID: {model_ID}", 
                n_mean=3, 
            )
        
        # .. to store results, 
        model_acc  = dict()
        loss_acc = dict()
        n          = 3 # use last 3 results in history, 
            
        # .. train & valid
        acc_results = pd.DataFrame(history.history).iloc[-n::,:].mean(axis=0)
        model_acc["model_acc_train"]    = acc_results.loc["acc"]  
        model_acc["model_acc_valid"]    = acc_results.loc["val_acc"]
        loss_acc["loss_acc_train"]  = acc_results.loc["loss"]
        loss_acc["loss_acc_valid"]  = acc_results.loc["val_loss"]             
        
        
        
        #### -------------------------------------------------
        #### STEP 5. LOAD TEST DATA and store all sort of data in the last loop !
        # ..... prep, 
        baseline_acc              = dict()
        one_model_predictions     = dict() # with predictions collected for each subset separately, 
        # .....
        test_subset_names_to_load = subset_names_to_load[2::]
        xy_names_test             = xy_names[2::]         
        # .....
        for ti, (one_test_xy_name, one_test_subset_name_to_load) in enumerate(zip(xy_names_test, test_subset_names_to_load)):

            test_id = ti+2    

            # generator for test data
            test_datagen  = ImageDataGenerator(rescale=1/255)
            
            # .. find how many images there are
            temp_testset  = test_datagen.flow_from_directory(
                os.path.join(dataset_path, subset_names_to_load[test_id]),
                batch_size     =params["batch_size"],
                target_size    =params["img_size"],
                shuffle        =False
                )
            
            # .. get all images in one batch, 
            test_ing_number = len(temp_testset.filenames)
            testset         = test_datagen.flow_from_directory(
                os.path.join(dataset_path, subset_names_to_load[test_id]),
                batch_size     =test_ing_number,
                target_size    =params["img_size"],
                shuffle        =False
                )  

        
            #### -------------------------------------------------
            #### STEP 5. COLLECT RESULTS

            #### calculate test set accuracy, 
            
            # .. get predictions (in dummy array)
            test_preds = model.predict_generator(testset)
            print(f'{one_test_xy_name} Predictions:', test_preds.shape)
            y_true     = testset.classes # array with true labels
            y_pred     = test_preds.argmax(axis=1) # array with predicted labels
        
            # ... calculate accuracy_score
            model_acc[f"model_acc_{one_test_xy_name}"]    = accuracy_score(y_true, y_pred)
            loss_acc[f"loss_acc_{one_test_xy_name}"]  = np.nan

            #### caulate test set baseline
            baseline_acc[f"baseline_acc_{one_test_xy_name}"] = pd.Series(y_true).value_counts(normalize=True).sort_values(ascending=True).iloc[0]
            
            #### store model predictions, 
            predictions                 = y_pred
            decoded_predictions         = pd.Series(y_pred).map(class_decoding).values
            model_predictions_proba     = test_preds
            decoded_y_labels            = pd.Series(y_true).map(class_decoding).values
            batch_labels_df             = pd.DataFrame(testset.filenames, columns=["imgname"])
            batch_labels_df["clasname"] = decoded_y_labels # miniversiton with 2 most importnat columns
            # ...
            one_model_predictions[one_test_xy_name] = {
                    "idx_in_batch":            np.arange(batch_labels_df.shape[0]),
                    "original_labels":         decoded_y_labels, 
                    "model_predictions":       decoded_predictions, 
                    "model_predictions_proba": model_predictions_proba,
                    "acc_restuls_and_params":  None, # will be added later on
                    "class_encoding":          class_encoding,
                    "class_decoding":          class_decoding,
                    "batch_labels_df":         batch_labels_df["clasname"],
                    "knn_kneighbors":          None
            }# added, in case I woudl have some dounbts about results origine, 

            

            
        #### -------------------------------------------------
        #### STEP 6.  Comllect more results, including lists that were created over each test subset
        
        # - 1 - collect acc_restuls_and_params
        acc_restuls_and_params = {
                         "model_ID": model_ID,
                         "run_name": run_name,
                         "method": method,
                         "dataset_name": dataset_name, 
                         "dataset_variant": dataset_variant,
                         "module": module_name,
                         "unit_test":False,
                         # ....
                         **baseline_acc,
                         **model_acc,
                         **loss_acc,       # nn only 
                         **params,
                         "pca_components_used":0
        }
        model_acc_and_parameters_list.append(acc_restuls_and_params) # in list, so it can be used as pd.df immediately, 
        print("\n----",acc_restuls_and_params,"\n\n----\n\n")
                         
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
 
        # - 4- store model predictions, 
        for one_test_xy_name in xy_names_test:    
            one_model_predictions[one_test_xy_name]['acc_restuls_and_params'] = acc_restuls_and_params
        # and finally, add this to the big dict wiht all the results, 
        model_predictions_dict[model_ID] = one_model_predictions 
                                  
    # ..................................................
    return model_acc_and_parameters_list, model_predictions_dict, model_parameters_dict, model_history_dict,


                        
            
 
                       