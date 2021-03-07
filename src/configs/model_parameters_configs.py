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


from sklearn.model_selection import ParameterGrid


# parameters used to train skilearn dn cnn tranfer learning models
MODEL_PARAMETERS_GRID = {
    
    "knn" : ParameterGrid([
        {     
            'method_group':   ["sklearn_models"],   # for all classical models, 
            'method_variant': ["no_pca"],       # eg: SVM has linear or rbf, or typically pca or not, (nothing)
            'n_neighbors':    list(range(2,10,2)),    # I do not use k==1, ! # I tried higher values and they were not very usefull
            'weights':        ['uniform','distance'], # Weighting function
            'p':              [2],                  # L1 and L2 distance metrics
            'pca':            [0],                  # is 0, no PCA applied, 
            'random_state_nr':[0],                  # in the list
            'pc': [['n_neighbors', 'weights', 'p']] # parameter names in the grid for classifier, (list in list)
        },
        {
            'method_group':   ["sklearn_models"],   # for all classical models, 
            'method_variant': ["pca"],          # eg: SVM has linear or rbf, or typically pca or not, (nothing)
            'n_neighbors':    list(range(2,10,2)),                  # I do not use k==1, ! # I tried higher values and they were not very usefull
            'weights':        ['uniform','distance'],          # Weighting function
            'p':              [2],                  # L1 and L2 distance metrics
            'pca':            [250],                # is 0, no PCA applied, 
            'random_state_nr':[0],                  # in the list
            'pc': [['n_neighbors', 'weights', 'p']] # parameter names in the grid for classifier, (list in list)  
        }
        ]), 
    
    
    'random_forest': ParameterGrid([
        {
            'method_group': ["sklearn_models"],     # for all classical models, 
            'method_variant': ["no_pca"],           # eg: SVM has linear or rbf, or typically pca or not, (nothing)
            'random_state_nr':[0],
            'max_depth':[4,5,6],
            'n_estimators': [10,25,50,100,150, 200],
            'class_weight': ['balanced'],
            'pca':[0],
            'pc':[['n_estimators', 'max_depth', 'class_weight']]
        },
        {
            'method_group': ["sklearn"],           # for all classical models, 
            'method_variant': ["pca"],           # eg: SVM has linear or rbf, or typically pca or not, (nothing)
            'random_state_nr':[0],
            'max_depth':[4,5,6],
            'n_estimators': [10,25,50,100,150, 200],
            'class_weight': ['balanced'],
            'pca':[30, 200],
            'pc':[['n_estimators', 'max_depth', 'class_weight']]
        }
       ]),
    
    "dense_nn" : ParameterGrid([{
            # ... for method classyficaiton 
            'method_group':   ["cnn_transfer_learning"],           # for all classical models, 
            'method_variant': ["two_layers_Adam"],                          # eg: SVM has linear or rbf, or typically pca or not, (nothing)
            'random_state_nr':[0],
            'pca':[0],
            # ...for cnn
            "model":["two_layers"],
            # ..
            "h1_unit_size":[36, 72, 144, 288],
            "h1_Dropout" : [0.5],
            "h1_activation": ["relu"],
            # ...
            "out_activation":["softmax"],
            "optimizer":["Adam"],
            "metrics": [["acc"]],
            # ...
            'fit__batch_size' : [16, 32, 64, 128],
            "EarlyStopping__patience": [3],
            "fit__epoch": [100]   
        }])
     
    }       