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
import pickle

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
        run_name,
        dataset_name,
        dataset_variant,
        module_name,
        
        # input data,
        cnn_model_function,
        grid, # list or list-like, obj created wiht ParameterGrid() 
        path, # path to data
        input_data_info_df=None, # df, with 3. columns, 
        unit_test = False,   
                         
        # options, 
        default_validation_split=0.2,  # float, used only if this value is not provided in input_data_info_df or validation_split pramatere in 
        datagen_params = dict(rescale =1/255), # dict, for keras imagedatagenerator, used only if not provided with parameter grid,     
                         
        # info, 
        verbose=False, # detailed info, 
        model_fit__verbose=0,                 
    ):
    """
        this function will train model provided with the  
    
        important:
        cnn_model_function  : this fucntion needs to return two objects 
                              * keras model : compileed already 
                              * list with callback function/s - always a list !
                              here is how teturn function looks:
                                return model, [callback_function]
    
        Limitations, 
        because we are creating valid dataset as subset of train data with image data generators, 
        these must be done using the same image generator (or else it will return Value Err after 1st epoch)
        from that reason, validaiton data have the same transformations as train data, and return typically lower acc values as 
        the tests done with other technicques and pielines, 
        
        
        code example:
        _ ,_ , _ , _ = deNovoCNN_gridsearch(        
                # model/run description 
                method             = ai_method,
                run_name           = "test_run",
                dataset_name       = dataset_name,
                dataset_variant    = dataset_variant,
                module_name        = None,

                # input data,
                cnn_model_function = create_cnn_model,
                grid               = grid,
                path               = path, # path to dir with data subsets in separate folders for train, test ect...
                input_data_info_df = CNN_DF_INFO, # df, with 3. columns, subset_role/results_name/dirname

                # results and info, 
                verbose=True
        )        
 
    """    
      
    
    # Set up, ............................................................................. 
       
    #.. variables,
    model_ID = -1   # to start usnique counts of the models from 0 
    colname_with_classname_in_batch_labels_table = "classname"
    
    #.. objects to store the results, 
    model_acc_and_parameters_list = list()
    model_predictions_dict        = dict()
    model_parameters_dict         = dict() # for iterations from grid
    model_history_dict            = dict()
    
    # create input_data_info_df is None available, 
    if input_data_info_df is None:
        input_data_info_df = pd.DataFrame([
            {
                 "subset_role":  "train",
                 "subset_results_name": "train",
                 "subset_dirname": "train"
            },
            {
                 "subset_role":  "valid",
                 "subset_results_name": "valid",
                 "subset_dirname": None     
            },
            {
                 "subset_role":  "test",
                 "subset_results_name": "test",
                 "subset_dirname": "test"     
            }
        ])    
    else:
        pass      

    #.. 
    if unit_test==True:
        subset_dirname_to_use_for_unit_test = input_data_info_df.subset_dirname.iloc[0]
        input_data_info_df.subset_dirname = subset_dirname_to_use_for_unit_test
    else:
        pass   
        
        
    # for loop for grid serch wiith each parameter combination,, ............................... 
    for params_i, params in enumerate(grid):
        model_ID +=1
        
        
        # check for required parameters, except for validation split, 
        try:
            params['method_group']
        except:
            params['method_group']="unknown"
        try:
            params['method_variant']
        except:
            params['method_variant']="unknown"
        try:
            params['random_state_nr']
        except:
            params['random_state_nr']=0 
        try:
            params['batch_size']
        except:
            params['batch_size']=10
        try:
            params['img_size']
        except:
            params['img_size']=[128, 128]
        try:
            params['epoch']
        except:
            params['epoch']=100
        try:
            params['early_strop']
        except:
            params['early_strop']=None
            
            
        # add or reset status column un info df, 
        input_data_info_df["status"]="awating"

        #.. plot history, 
        '''
            printed here, so you know what data are being loaded with image generators, 
        '''
        if verbose==True:
            print(f"\n\n..................................................")
            print(f"model_ID: {model_ID}")
            print(f"method_group: {params['method_group']}")
            print(f"method: {method}")
            print(f"method_variant: {params['method_variant']}")
            print(f"............................... input data ...")
            print(f"run_name: {run_name}")
            print(f"dataset_name: {dataset_name}")
            print(f"dataset_variant: {dataset_variant}")
            print(f"unit_test: {unit_test}")
            print(f"............................... basic params ...")
            print(f"random_state_nr: {params['random_state_nr']}")            
            print(f"batch_size: {params['batch_size']}") 
            print(f"img_size: {params['img_size']}") 
            print(f"epoch: {params['epoch']}") 
        else:
            pass        
   
        # set parameters for 
        try:
            train_datagen_params = params["train_datagen_params"]
            valid_datagen_params = params["valid_datagen_params"]
            test_datagen_params  = params["test_datagen_params"]
            datagen_params_info = "imgadatagen prams provided by the user"
        except:
            train_datagen_params = datagen_params
            valid_datagen_params = datagen_params
            test_datagen_params = datagen_params   
            datagen_params_info = "using default imgadatagen params"
        
        # load train & valid data
        if unit_test==True:
            valid_datagen_params = train_datagen_params
            test_datagen_params  = train_datagen_params        
        else:
            pass    
        
        # dirnames (only one dirname is allowed with generaqtors,)
        train_subset_dirname = input_data_info_df.subset_dirname.loc[input_data_info_df.subset_role=="train"].iloc[0]
        valid_subset_dirname = input_data_info_df.subset_dirname.loc[input_data_info_df.subset_role=="valid"].iloc[0] # not used at this moment, 

    
        # OPTION 1, subset valid data from train data 
        if valid_subset_dirname is None or isinstance(valid_subset_dirname, float):
            # set-up directory names and datagen parameters, 
            if isinstance(valid_subset_dirname, float):
                train_datagen_params["validation_split"]=valid_subset_dirname
                
            # else its None, so we have to get validaiton split value from somewhere, 
            else:
                try:
                    train_datagen_params["validation_split"] #KeyError if it is missing, 
                except:
                    train_datagen_params["validation_split"]=default_validation_split
                
            train_datagen  = ImageDataGenerator(**train_datagen_params)
            TRAINING_DIR   = os.path.join(path, train_subset_dirname)
            VALIDATION_DIR = TRAINING_DIR
            
            # update, status:
            input_data_info_df.loc[input_data_info_df.subset_role=="train", "status"] = "Loading"
            input_data_info_df.loc[input_data_info_df.subset_role=="valid", "status"] = f"{train_datagen_params['validation_split']} of train"

            #.. for train dataset
            trainset = train_datagen.flow_from_directory(
                TRAINING_DIR, 
                batch_size     =params["batch_size"], 
                target_size    =params["img_size"],
                shuffle        =True,
                subset         ="training"
            )

            #.. for validation data, no shuffle, made from the same dataset as train data, 
            validset = train_datagen.flow_from_directory(
                VALIDATION_DIR, 
                batch_size     =params["batch_size"], 
                target_size    =params["img_size"],
                shuffle        =False,
                subset         ="validation"
            )
    
        # OPTION 2, valid data are loaded from separate directory,           
        else:           
            #.. create image generator, validation_split=params["validation_split"]
            train_datagen  = ImageDataGenerator(**train_datagen_params)
            valid_datagen  = ImageDataGenerator(**valid_datagen_params)     
            
            TRAINING_DIR   = os.path.join(path, train_subset_dirname)
            VALIDATION_DIR = os.path.join(path, valid_subset_dirname)
            
            # update, status:
            input_data_info_df.loc[input_data_info_df.subset_role=="train", "status"] = "Loading"
            input_data_info_df.loc[input_data_info_df.subset_role=="valid", "status"] = "Loading"

            #.. for train dataset
            trainset = train_datagen.flow_from_directory(
                TRAINING_DIR, 
                batch_size     =params["batch_size"], 
                target_size    =params["img_size"],
                shuffle        =True
            )

            #.. for validation data, no shuffle, made from the same dataset as train data, 
            validset = valid_datagen.flow_from_directory(
                VALIDATION_DIR, 
                batch_size     =params["batch_size"], 
                target_size    =params["img_size"],
                shuffle        =False
            )


        # Collect class encoding/decoding - its not the standard one, .....................
        class_encoding  = trainset.class_indices
        class_decoding  = dict(zip(list(class_encoding.values()),list(class_encoding.keys())))

        
        
        # add some info:
        if verbose==True:
            print(f"............................... input data - info df ...")
            print(input_data_info_df)
            print(f"..................................................\n")
        else:
            pass        

    
        
        # train the model, collect the results, and plot history, .........................
        
        #.. create the model,  
        model, callback_function =  cnn_model_function(
            input_size=(params["img_size"][0], params["img_size"][1], 3), 
            output_size=trainset.num_classes, 
            params=params,
            verbose=verbose
        )
          
        #.. train the model
        if callback_function is not None:
            history = model.fit_generator(
                generator        =trainset,          # you provide iterators, instead of data, 
                validation_data  =validset, 
                epochs           =params["epoch"], 
                callbacks        =callback_function,   # LIST, functions that can be applied at diffeerent stages to fight overfitting, 
                verbose          =model_fit__verbose
            )
        else:
            history = model.fit_generator(
                generator        =trainset,          # you provide iterators, instead of data, 
                validation_data  =validset, 
                epochs           =params["epoch"], 
                verbose          =model_fit__verbose
            )            
            
        #.. store the results,
        model_acc   = dict()
        model_loss  = dict()
        n           = 3 # use last 3 results in history, 
        acc_results = pd.DataFrame(history.history).iloc[-n::,:].mean(axis=0)
        model_acc["model_acc_train"]      = acc_results.loc["acc"]  
        model_acc["model_acc_valid"]      = acc_results.loc["val_acc"]
        model_loss["loss_acc_train"]      = acc_results.loc["loss"]
        model_loss["loss_acc_valid"]      = acc_results.loc["val_loss"]             
          
 
        # LOAD TEST DATA and store all sort of data in the last loop !, .........................

        # prepare objects to store results,  
        baseline_acc              = dict()
        one_model_predictions     = dict() # with predictions collected for each subset separately, 
        # it will be used temporaly, untul the last chunk of code to reload model predictions to final location

        
        # first, chek if test data were provided, 
        if (input_data_info_df.subset_role=="test").sum()==0:
            pass
        
        else:
            # get df subset with test data 
            test__input_data_info_df = pd.DataFrame(input_data_info_df.loc[input_data_info_df.subset_role=="test",:])
            test__input_data_info_df.reset_index(inplace=True, drop=True)
            # now check if there is anythign to load, and pass if None, 
            if test__input_data_info_df.shape[0]==1 and test__input_data_info_df.subset_dirname.iloc[0] is None:
                pass
            else:    
                
                # loop over each row, to load and evaluate each test data
                if verbose==True:
                    print(f"generating predicitons for test data: {test__input_data_info_df.shape[0]} subsets")
                else:
                    pass
                for test_subset_nr in range(test__input_data_info_df.shape[0]):

                    # load the dataset, 
                    one_test_xy_name             = test__input_data_info_df.subset_results_name.iloc[test_subset_nr]
                    one_test_subset_name_to_load = test__input_data_info_df.subset_dirname.iloc[test_subset_nr] 
                
                    #.. generator for test data
                    test_datagen  = ImageDataGenerator(**test_datagen_params)

                    #.. find how many images there are
                    temp_testset  = test_datagen.flow_from_directory(
                        os.path.join(path, one_test_subset_name_to_load),
                        batch_size     =params["batch_size"],
                        target_size    =params["img_size"],
                        shuffle        =False
                        )

                    #.. get all images in one batch, 
                    test_img_number = len(temp_testset.filenames)
                    testset         = test_datagen.flow_from_directory(
                        os.path.join(path, one_test_subset_name_to_load),
                        batch_size     =test_img_number,
                        target_size    =params["img_size"],
                        shuffle        =False
                        )  

                    # calculate test set accuracy, 

                    #.. get predictions (in dummy array)
                    test_preds = model.predict_generator(testset)
                    y_true     = testset.classes # array with true labels
                    y_pred     = test_preds.argmax(axis=1) # array with predicted labels
                    model_acc[f"model_acc_{one_test_xy_name}"]  = accuracy_score(y_true, y_pred)
                    model_loss[f"model_loss_{one_test_xy_name}"]= np.nan

                    #.. caulate test set baseline
                    baseline_acc[f"baseline_acc_{one_test_xy_name}"] = pd.Series(y_true).value_counts(normalize=True).sort_values(ascending=True).iloc[0]

                    #.. store model predictions, 
                    predictions                 = y_pred
                    decoded_predictions         = pd.Series(y_pred).map(class_decoding).values
                    model_predictions_proba     = test_preds
                    decoded_y_labels            = pd.Series(y_true).map(class_decoding).values
                    batch_labels_df             = pd.DataFrame(testset.filenames, columns=["imgname"])
                    batch_labels_df["clasname"] = decoded_y_labels # miniversiton with 2 most importnat columns
                    #..
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

            

        # Collect more results, including lists that were created over each test subset..................            

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
                         **model_loss,       # nn only 
                         **params,
                         "pca_components_used":0
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
 
        # - 4- add acc_restuls_and_params to each model prediction, and store them in the dict with model ID 
        for one_test_xy_name in list(one_model_predictions.keys()):    
            one_model_predictions[one_test_xy_name]['acc_restuls_and_params'] = acc_restuls_and_params
        
        # and finally, add this to the big dict wiht all the results, 
        model_predictions_dict[model_ID] = one_model_predictions 

        
        
        
        # Report on results, .....................................................................
        
        #.. plot history, 
        if verbose==True:
            plot_NN_loss_acc(
                model_history_df=pd.DataFrame(history.history), 
                title=f"model_ID: {model_ID}\n{model_acc}", 
                n_mean=3, 
            )
        else:
            pass        
        
        
    # ..................................................
    return model_acc_and_parameters_list, model_predictions_dict, model_parameters_dict, model_history_dict







# Function, .................................................
def train_denovoCNN_models_iteratively(*,
                                      
    # names  
    run_ID,            # str, Unique ID added to name of each file to allow running and saving similar module create with different parameters or different data
    dataset_name,      # str, global, provided by the wrapper function,
    dataset_variant,   # str, global, provided by the wrapper function,
    module_name,       # str, global, provided by the wrapper function,                                  
                  
    # define input data,
    path,                          # str, or list of the same lengh and order as subset_collection_names, path to input data
    subset_collection_names,       # list, anything that will allow you to identify which subset and run you are using             
    input_data_info_df_dict,       # dict, with input_data_info_df, each is df, with 3. columns, subset_role/results_name/dirname
                                                                 
    # model parameters,                   
    method_name,                   # str, keywod in the function {knn, svm, logreg, dt, rf}
    cnn_model_function,            # function providing sequencian keras model, and list with 
    datagen_params = dict(rescale =1/255), # default datagenerator parameters applied to all data
    grid,                          # ParameterGrid object, wiht parameters for a given function,                
                                                                   
    # model selection cycles,                       
    models_selected_at_each_cycle         = 0.3,              # int, how many models with best performace will be selected and trained with another round of training and with next subset collection 
    include_method_variant_with_selection = True,             # bool, if True, top models_selected_at_each_cycle wiht different model variant will be selected to next cycle
    include_random_nr_with_selection      = False,            # bool, if True, top models_selected_at_each_cycle wiht different random nr will be selected to next cycle
    sort_models_by                        = "model_acc_valid",# str {"model_acc_valid"} 
    find_min_score                        = False,            # if false, the scores will be used in descending order, 
                                      
    # saving and other, 
    save_path,                     # str, eg PATH_results from configs 
    save_partial_results=True,     # bool, if True, it will save model results at each iteration                            
    valid_proportion=0.2,  # used only if validation datasets is not specified, 
    unit_test=False, 
    model_fit__verbose=0,
    verbose=False
):

 

    # create path for results wiht all options, 
    path_results = os.path.join(save_path, f"{method_name}__{dataset_name}__{dataset_variant}")
    try: 
        os.mkdir(path_results)
        if verbose==True:
            print(f"\n Crated: {path_results}\n")
        else:
            pass
    except: 
        pass
    

    
    # ** / grid search       
    run_names = list()
    for cycle_nr, subset_collection_name in enumerate(subset_collection_names):
        
        
        print(f"\n - - - CYCLE: {cycle_nr} - - -\n")
        
        # ...............................................
        # cycle set up
        input_data_info_df = input_data_info_df_dict[subset_collection_name]
        
        # set unique run name, 
        run_name = f"{subset_collection_name}__{run_ID}"
        run_names.append(run_name)
        
        # set name added to each saved file wiht results from that cycle
        file_name_fingerprint = f'{method_name}__{dataset_name}__{dataset_variant}__{module_name}__{run_name}'
        
        # path to data
        'if str, one path was given, if list, datasets for different cycles awere located on different paths'
        if isinstance(path, str):
            path_to_data = path
        else:
            path_to_data = path[cycle_nr]
        
        
        # ...............................................
        # find grid with parameters
        
        if cycle_nr==0:
            "the grid is provided externally in 0 cycle"
            cycle_grid = grid
            
        else:
            "here you must collect parameters from the best performing models, and extract params for top nr of them"
            "options to include model variant in selection"
            sort_by = "model_acc_valid"
            

            # collect features you want to use to sort model results and get top of each of them
            features_used_to_group_models = ["method", "dataset_name", "dataset_variant", "module"]
            if include_random_nr_with_selection==True:
                features_used_to_group_models.append("random_state_nr")
            else:
                pass
            if include_method_variant_with_selection:
                features_used_to_group_models.append("method_variant")
            else:
                pass    

            # add these features to df, with the model results as one column
            for fi, feature in enumerate(features_used_to_group_models):
                if fi==0:
                    composite_feature = results_from_last_cycle.loc[:, feature].values.tolist()
                else:
                    composite_feature = [f"{x}__{y}" for (x,y) in zip(composite_feature, 
                                        results_from_last_cycle.loc[:, feature].values.tolist())]
            results_from_last_cycle["method_full_name"] = composite_feature
            

            
            # find best performing models in each group and sort them    
            method_groups    = results_from_last_cycle.method_full_name.unique().tolist()
            best_methods_IDs = list()
            for ii, mg in enumerate(method_groups):

                # subset summary_df for each method group
                df_subset =  results_from_last_cycle.loc[ results_from_last_cycle.method_full_name==mg, :]
                df_subset = df_subset.sort_values(sort_by, ascending=find_min_score)
                df_subset.reset_index(inplace=True, drop=True)

                # find how many models will be selected for the next cycle,
                if models_selected_at_each_cycle<1 and models_selected_at_each_cycle>0:
                    mnr = int(np.ceil(df_subset.shape[0]*models_selected_at_each_cycle))
                    
                elif models_selected_at_each_cycle==0:
                    mnr = 1
                    
                else:
                    mnr = models_selected_at_each_cycle

                # because I had some rare situations with problems,
                if mnr==0:
                    mnr=1
                else:
                    pass    

                # find top n models in each
                best_methods_IDs.extend(df_subset.model_ID.values[0:mnr].tolist()) #this will extend the list by each nr of id numbers
                      
            # create new grid
            cycle_grid=list()
            for gidx in best_methods_IDs:
                cycle_grid.append(model_parameter_list[gidx]['params']) # yes its 'para  ms'! its a mistake, that I have no time to correct 

                
        # train models 
        results_list, model_predictions_dict, model_parameter_list, model_history_dict = deNovoCNN_gridsearch(        
                # model/run description 
                method             = method_name,
                run_name           = run_name,
                dataset_name       = dataset_name,
                dataset_variant    = dataset_variant,
                module_name        = module_name,

                # input data,
                path               = path_to_data, # path to dir with data subsets in separate folders for train, test ect...
                input_data_info_df = input_data_info_df, # df, with 3. columns, subset_role/results_name/dirname
                datagen_params     = datagen_params, # default datagenerator parameters applied to all data
                
                # model and params,
                cnn_model_function = cnn_model_function,
                grid               = cycle_grid,
                
                # options, 
                default_validation_split = valid_proportion,
                unit_test                = unit_test,

                # results and info,
                model_fit__verbose       = model_fit__verbose,
                verbose                  = True
            )        

        # this is for the next cylce
        results_from_last_cycle = pd.DataFrame(results_list)


        # ** / save the results, 
        if save_partial_results==True:
            save_te_results=True
        else:
            if cycle_nr==(len(subset_collection_names)-1):
                save_te_results=True
            else:
                save_te_results=False
            
            
        # create path for results
        if save_te_results==True:
            os.chdir(path_results)

            if verbose==True:
                print(f"The results will be saved with as:\n{file_name_fingerprint}")
            else:
                pass

            # save results and metadata on each model, 
            pd.DataFrame(results_list).to_csv(f"{file_name_fingerprint}__summary_table.csv", header=True)

            # save model predictions, 
            with open(f"{file_name_fingerprint}__model_predictions_dict.p", 'wb') as file: # wb - write binary,
                pickle.dump(model_predictions_dict, file) 

            # save model parameters to re-run the models
            with open(f"{file_name_fingerprint}__model_parameters_list.p", 'wb') as file: # wb - write binary,
                pickle.dump(model_parameter_list, file) 
                

            # save history_dict to re-run the models - speciffic for tf models 
            with open(f"{file_name_fingerprint}__model_history_dict.p", 'wb') as file: # wb - write binary,
                pickle.dump(model_history_dict, file)                 
                
                
        else:
            if verbose==True:
                print(f"The results for this cycle were not saved, only final results are going to be saved")
            else:
                pass            
            pass