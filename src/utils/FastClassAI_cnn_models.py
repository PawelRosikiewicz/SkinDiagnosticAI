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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier # accepts only numerical data
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
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





# Function, ...........................................................................
def TwoConv_OneNN_model(*, input_size, output_size, params=dict(), verbose=False):
    '''
        creeast simple sequential model
        params : dict with selected parameters and options used for model creation, 
        
        input_size, 
        output_size, 
        params=dict(), 
        verbose=False
        
    '''

    # default params, 
    try:
        params["f1_units"]
    except:
        params["f1_units"]=100
    try:
        params["f1_dropout"]
    except:
        params["f1_dropout"]=0            
    try:
        params["optimizer"]
    except:
        params["optimizer"]="Adam"     
    try:
        params['early_strop']
    except:
        params['early_strop']=None  

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

    if verbose==True:
        model.summary()    
    else:
        pass    
        
    # create callback function, 
    if params['early_strop'] is not None:
        callback_function = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["early_strop"])
        return model, [callback_function]
    else:
        return model, None




