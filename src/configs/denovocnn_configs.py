# ********************************************************************************** #
#                                                                                    #
#   Project: SkinAnaliticAI                                                          #                                                         
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz_gmail.com                                                  #
#                                                                                    #
#.  This notebook is a part of Skin AanaliticAI development kit, created             #
#.  for evaluation of public datasets used for skin cancer detection with            #
#.  large number of AI models and data preparation pipelines.                        #
#                                                                                    #     
#   License: MIT                                                                     #
#.  Copyright (C) 2021.01.30 Pawel Rosikiewicz                                       #
#   https://opensource.org/licenses/MIT                                              # 
#                                                                                    #
# ********************************************************************************** #


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid



# configs, ...................................................................
'''
    dataset descryption adapted to my fucntions on iteratively running denovo cnn functions, 

'''
DATASETS_DESCRIPTION = {
        "small_dataset" : pd.DataFrame([
            {
                 "subset_role":  "train",
                 "subset_results_name": "train",
                 "subset_dirname": "trainvalid_small_cnn"
            },
            {
                 "subset_role":  "valid",
                 "subset_results_name": "valid",
                 "subset_dirname": 0.2
            },
            {
                 "subset_role":  "test",
                 "subset_results_name": "test",
                 "subset_dirname": "test_01"     
            },
            {
                 "subset_role":  "test",
                 "subset_results_name": "test_2",
                 "subset_dirname": "test_02"     
            }
        ]),           
        "large_dataset" : pd.DataFrame([
            {
                 "subset_role":  "train",
                 "subset_results_name": "train",
                 "subset_dirname": "trainvalid_all_cnn"
            },
            {
                 "subset_role":  "valid",
                 "subset_results_name": "valid",
                 "subset_dirname": 0.1
            },
            {
                 "subset_role":  "test",
                 "subset_results_name": "test",
                 "subset_dirname": "test_01"     
            },
            {
                 "subset_role":  "test",
                 "subset_results_name": "test_2",
                 "subset_dirname": "test_02"     
            }
        ])        
    }



# configs, ...................................................................
'''
    dataset descryption adapted to my fucntions on iteratively running denovo cnn functions, 

'''
train_datagen_params = dict(
        rescale =1/255,
        height_shift_range=0.1,
        width_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        rotation_range=30,
        brightness_range=[0.2,1.8], 
        zoom_range=[0.75,1.25], # ±25%
        channel_shift_range=.1,
        validation_split=0.3   ###################### important           
    )
valid_datagen_params = dict(rescale =1/255)
test_datagen_params = dict(rescale =1/255)


MODEL_PARAMETERS_GRID = {
    "cnn" : ParameterGrid([
        {   
            # parameters for keras image generators,
            'train_datagen_params': [train_datagen_params],
            'valid_datagen_params': [train_datagen_params],
            'test_datagen_params': [train_datagen_params],
            # input data and model names, 
            'method_group':   ["denovo_cnn"],      # for all classical models, 
            'method_variant': ['res_64_pixels'],    # eg: SVM has linear or rbf, or typically pca or not, (nothing)
            'random_state_nr':[0],                  # in the list
            #'pc': [['n_neighbors', 'weights', 'p']] # unused with cnn
            "batch_size": [16],
            "img_size":  [(200, 200)],
            # ........................
            "f1_units":  [100],
            "f1_dropout":[0],
            "optimizer": ["Adam"],
            # ........................
            "epoch": [2],
            "early_strop": [3]   
        }
        ]) 
    }
