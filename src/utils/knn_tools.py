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


from src.utils.image_augmentation import * # to create batch_labels files, 
from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch
from PIL import Image, ImageDraw
from matplotlib.font_manager import FontProperties
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier



# Function, ............................................................................................

def perfrom_grid_search(*, X, y, train_proportion=0.7, pipe, grid,  method_name=np.nan, verbose=False):

    
    # check the data, ................................ 
    assert type(X)==np.ndarray, "Incorrect obj type" # Test input df,
    assert type(y)==np.ndarray, "Incorrect obj type" # Test input df,
    

    # Data preparation, ...............................

    # .. Split data into train/test sets
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, 
        train_size=train_proportion, 
        test_size=(1-train_proportion),
        random_state=0
    )

    # .. test dimensions, 
    if verbose==True:
        print('Number of combinations:', len(grid))
        print("Input Data shapes are:", "train=",X_tr.shape," test=",X_te.shape)
    else:
        pass

    
    # Grid Search, ...............................

    # Save accuracy on test set
    test_scores = []

    # Enumerate combinations starting from 1
    for i, params_dict in enumerate(grid, 1):


        if verbose==True:
            # Print progress
            if i-1==0:
                print(f"GridSearch: ", end="")
            if i>1 and i<len(grid)-1:
                print(".",end="")
            if i==len(grid):
                print(".", end="\n")
        else:
            pass

        # Set parameters
        pipe.set_params(**params_dict)

        # Fit a k-NN classifier
        pipe.fit(X_tr, y_tr)

        # Save accuracy on test set
        params_dict['train_accuracy'] = pipe.score(X_tr, y_tr)
        params_dict['test_accuracy'] = pipe.score(X_te, y_te)
        params_dict['method'] = method_name

        # Save result
        test_scores.append(params_dict)

    if verbose==True:
        print('done')
    else:
        pass
    
    # prepare the results, ...................
    scores_df = pd.DataFrame(test_scores)
    return scores_df
  
  
  
  
  
  
  
  
# Function, ...........................................................................................................
def knn_grid_search(*, 
      method_name="knn", 
      path, 
      dataset_name, 
      subset_names_tr, 
      subset_names_te, 
      module_names, 
      class_encoding, 
      grid, 
      param_names_for_Classifier, 
      train_proportion=0.7, 
      random_state_nr=0, 
      store_predictions=True, 
      verbose=False,
      track_progresss=False                   
):

    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function,         Custom function that perfomes grid search using decision trees, on features extracted 
                            from images with different tf.hub modules. 
                            
                            
                            Optionally, it allows using pca, for tranforming 
                            extracted features intro selected number of principial components, 
                            later on used by SVM algorithm
                            
        # Inputs,     
          .................................................................................................
        . path.              : str, path to directory, with data sstored, 
        . dataset_name       : str, datassets name, used while creating          
        . logfile_name       : str, path to logfile
        . dataset_name       : 
        . subset_names_tr    : list, eg: [train", "valid"], these two dastasets will be concastenated in that order
                             Ussed exclu
        . subset_names_te    : list, eg: ["test"], these two dastasets will be concastenated in that order
                              Caution, I assumed that, more then one subset of data is keept in dataset file folder, ¨
                              eg, that you stored test and train data separately, 
        . module_names       : list, with names given to different moduless or methods used for feature extractio 
                             from images, 
        . param_names_for_DecisionTreeClassifier : list, with parameters that will be used exlusively, 
                                                 for DecisionTreeClassifier()
        . grid               : ParameterGrid() object, with parameters for DecisionTreeClassifier() and number 
                             of principial axes ussed instead of extracted features, 
                             eg:
                            grid = ParameterGrid({
                                  'criterion': ["gini"],   
                                  'max_depth': [3,5],
                                  'class_weight': ['balanced'],
                                  'pca':[0, 10, 30]})           # pca will not be used, or the alg, 
                                                                  will use either 10 or 30 principial 
                                                                  components to train decision tree
                         
         . store_predictions : bool, if True, predictions for all models, with train, validations and test datasets 
                               will be perfomed and stored in  model_predictions_dict
                              
         . class_encoding    : dict, key:<orriginal class name>:value<numerical value used by decision tre>
                               eg: dict(zip(list(class_colors.keys()), list(range(len(class_colors)))))
         . random_state_nr   : int, random state nr, used by sample split, and decision tree alg, 
         . train_proportion  : propotion of samples in inpur data for training, 
        
        # Returns,     
          .................................................................................................
          
          
          . model_acc_and_parameters_list : list, where each entry is a dict, with accuracy and parameters usied to build 
                                    a given model, and model_ID that can be used to retrieve items from two other 
                                    objectes returned by this funciotn, 
                                          
          . dot_data_dict          : dict, key=model_ID, stores decission trees in dot_data file format, 
                                     created using export_graphviz() for each model,
          
          . model_predictions_dict : dict, key=model_ID ()
                                     content: another dict, with "train, test and valid" keys 
                                            representing predictions made with eahc of these subsets
                                     each of them is also a dict. with, 
                                      >  "idx_in_used_batch":       index of each image in original img_batch 
                                                                    as created, using, subset_names_tr, and 
                                                                    load_encoded_imgbatch_using_logfile() function
                                      >  "original_labels":         array, with original class names for each image in a given dataset
                                      >  "model_predictions":       array, with preducted class names for each image in a given dataset
                                      >  "acc_restuls_and_params":  contains dict, with acc_restuls_and_params         
                                                                    to ensure reproducibility, 

        # Notes,     
          .................................................................................................                                                                
          I DO NOT SAVE MODELS, BECAUSE THESE TAKE A LOT OF MEMORY, THAT IS REALLY RESTRICTED ON MY COMPUTER, 
          MOREVER, KNN MODELS CARRIES ENTIRE INPUT DATASET IN IT !
          
          in case you wish to save models use joblib library 
          or visit: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

    """

    # dist to store results, 
    dot_data_dict = dict() # decision trees stored in dot format, 
    model_acc_and_parameters_list = list()
    model_predictions_dict = dict()
    # ...
    class_decoding = dict(zip(list(list(class_encoding.values())), list(class_encoding.keys()))) # reverse on class_encoding, 
    
    # ...
    model_ID = -1 # id number for each model, its predictions, I started with -1 so the first id will be 0 !
    for i, module_name in enumerate(module_names):
        """
            Note: I decided to load the data and tranform them at each iteration, 
                  because I was working with relatively small datasets, and it was easier, 
                  otherwise i woudl recommend to create a copy of inpout for models, and modify it with pca,
                  instead of relading entire dataset. 

            Note: I am evaluating each model with the same set of X valid and X te, because it was a goal, 
                  of that task, and only onbce, becuase it was exploratory data analysis, 
        """

        if track_progresss==True:
            print(f"{i} {module_name} _________________________________________ {pd.to_datetime('now')}")
        else:
            pass
        
        
        # Grid search, 
        for params in grid:    

            # PARAMETERS, ...................................
            model_ID +=1
            pca_axes_nr = params["pca"]
            dt_params_dct = dict(zip(param_names_for_Classifier,[params[x] for x in param_names_for_Classifier]))
            # ...
            Xy_names = ["train", "valid", "test"]
            
            if track_progresss==True:
                print('.', end="")
            else:
                pass

            # DATA PREPARATION,..............................

            # .................
            # load and ecode X,y arrays 

            # find any logfile created while saving img files, 
            os.chdir(path)
            logfiles = []
            for file in glob.glob(f"{''.join([module_name,'_',dataset_name])}*_logfile.csv"):
                logfiles.append(file)
                
                
            # ... info, 
            if verbose==True:
                print(f'{"".join(["."]*80)}')
                print(f'{module_name}, logfie: {logfiles[0]}')
                print(f" --- dt  params: {dt_params_dct}")
                print(f" --- pca params: {pca_axes_nr}")
            else:
                pass

            # train data, 
            X, batch_labels = load_encoded_imgbatch_using_logfile(logfile_name=logfiles[0], load_datasetnames=subset_names_tr)
            X = X.astype(np.float)
            y = pd.Series(batch_labels.classname).map(class_encoding).values.astype("int")

            # ... Split data into train/test sets
            X_tr, X_valid, y_tr, y_valid = train_test_split(
                X, y, 
                train_size=train_proportion, 
                test_size=(1-train_proportion),
                random_state=random_state_nr 
            )
            # ... get xy_idx to identify raw images in train/valid datasets, 
            _, _, idx_y_tr, idx_y_valid = train_test_split(
                X, np.arange(X.shape[0], dtype="int"), 
                train_size=train_proportion, 
                test_size=(1-train_proportion),
                random_state=random_state_nr    # Caution, random_state_nr must be the same as in the above, 
            )
            
            # test data, 
            X_te, batch_labels = load_encoded_imgbatch_using_logfile(logfile_name=logfiles[0], load_datasetnames=subset_names_te)
            X_te = X_te.astype(np.float)
            y_te = pd.Series(batch_labels.classname).map(class_encoding).values.astype("int")
            idx_y_te = np.arange(y_te.shape[0], dtype="int")
            
            # place all in dict,
            X_dct = dict(zip(Xy_names, [X_tr, X_valid, X_te]))
            y_dct = dict(zip(Xy_names, [y_tr, y_valid, y_te]))
            idx_y_dct = dict(zip(Xy_names, [idx_y_tr, idx_y_valid, idx_y_te]))


            # ...................,..............................
            # perfomr pca, 
            if pca_axes_nr!=0:

                # Train PCA model and use it to tranfomr data
                pca = PCA(n_components=pca_axes_nr) # it will use max nr of components == nr of features in dataset !
                pca.fit(X_tr, y=None) # Unsupervised learning, no y variable
                # ...
                for xyname in Xy_names:
                    X_dct[xyname] = pca.transform(X_dct[xyname])
            else:
                pass

            
            
            # MODEL ,....................................................
            
            # ...................
            # Create Most frequet baseline, 
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

            
            # ..................
            # Create classifier in the pipe, 
            model = Pipeline([
                ('scaler', StandardScaler()),      # With standardization
                #('scaler', None),                 # Better performance without standardization! in grid, 
                ('knn', KNeighborsClassifier(
                    algorithm='brute',             # Brute-force search
                    n_jobs=-1                      # As many parallel jobs as possible
                ))
            ])
   
            # Set parameters
            model.set_params(**dt_params_dct)

            # Fit a k-NN classifier
            model.fit(X_dct["train"], y_dct["train"])

            # get accuracy,
            model_acc = dict()
            for xyname in Xy_names:
                model_acc[f"model_acc_{xyname}"] = model.score(X_dct[xyname], y_dct[xyname])

            if verbose==True:
                print(" --- ", model_ID, model_acc)    
            else:
                pass
            

            
            # COLLECT THE RESULTS ,..............................  
            'acc_restuls_and_params were added to all objects in case I woudl have some dounbts about results origine,'


            # collect acc_restuls_and_params
            acc_restuls_and_params = {
                "random_state_nr": random_state_nr,
                 "model_ID": model_ID,
                 "method": method_name,
                 "module": module_name,
                 **baseline_acc,
                 **model_acc,
                 **dt_params_dct,
                 "pca_components_used":pca_axes_nr,
            }
            model_acc_and_parameters_list.append(acc_restuls_and_params) # in list, so it can be used as pd.df immediately, 
                
            # Collect Model predictions, 
            if store_predictions==True:
                one_model_predictions = dict()
                for xyname in Xy_names:
                    # make predictions and decode them,
                    predictions         = model.predict(X_dct[xyname])
                    decoded_predictions = pd.Series(predictions).map(class_decoding).values
                    model_predictions_proba = model.predict_proba(X_dct[xyname])
                    decoded_y_labels    = pd.Series(y_dct[xyname]).map(class_decoding).values
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

        if track_progresss==True:
            print(f"\nDONE _________________________________________ {pd.to_datetime('now')}",end="\n\n")
        else:
            pass
        
    # ..................................................
    return model_acc_and_parameters_list, model_predictions_dict
            
 