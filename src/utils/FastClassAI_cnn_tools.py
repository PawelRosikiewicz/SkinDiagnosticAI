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
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt # for making plots, 

# project functions
from src.utils.FastClassAI_skilearn_tools import prepare_list_with_subset_collection_composition_list
from src.utils.cnn_transfer_learning_tools import CNN_GridSearch
from src.utils.cnn_transfer_learning_tools import create_keras_two_layer_dense_model
from src.utils.cnn_transfer_learning_tools import plot_NN_loss_acc



# Function, .................................................
def train_and_test_cnn_tranfer_learning_models(*,

    # names  
    run_ID,            # str, Unique ID added to name of each file to allow running and saving similar module create with different parameters or different data
    dataset_name,      # str, global, provided by the wrapper function,
    dataset_variant,   # str, global, provided by the wrapper function,
    module_name,       # str, global, provided by the wrapper function,                                  
                  
    # define input data,
    subset_collection_names,            # list, anything that will allow you to identify which subset and run you are using             
    subset_collection_composition_dict, # list, wiht dataframes, the same lenght as subset_collection_names, returned by prepare_list_with_subset_collection_composition_list
    data_subsets_role,                  # dict, with names of subsets used as train, valid and test + in case you wish to use
 
    # model parameters,
    method_name,       # str, keywod in the function {knn, svm, logreg, dt, rf}
    grid,              # ParameterGrid object, wiht parameters for a given function, 
            
    # model selection cycles,                       
    models_selected_at_each_cycle = 0.3,          # int, how many models with best performace will be selected and trained with another round of training and with next subset collection 
    include_method_variant_with_selection = True, # bool, if True, top models_selected_at_each_cycle wiht different model variant will be selected to next cycle
    include_random_nr_with_selection = False,     # bool, if True, top models_selected_at_each_cycle wiht different random nr will be selected to next cycle
    sort_models_by = "model_acc_valid",           # str {"model_acc_valid"} 
                                  
    # saving
    save_path,             # str, eg PATH_results from configs 
    save_partial_results=True, # bool, if True, it will save model results at each iteration
                                  
    # pipe variables, # same for all cycles,                              
    class_encoding,        # dict, global key: class_name, value:int 
    class_decoding,        # dict, global key: int, value:class_name
    train_proportion=0.7,  # used only if validation datasets is not specified, 
    dropout_value=None,    # str, from configs, 
    unit_test=False,
  
    # other, 
    plot_history=True, # plots accuracy and error over epohs for each nn
    verbose=True
):

    
    # ** / options 
    if verbose==True:
        display_partial_results=True
    else:
        display_partial_results=False

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
        # get df with path and filenames to load
        subset_collection_composition_df = subset_collection_composition_dict[subset_collection_name]
        
        # set unique run name, 
        run_name = f"{subset_collection_name}__{run_ID}"
        run_names.append(run_name)
        
        # set name added to each saved file wiht results from that cycle
        file_name_fingerprint = f'{method_name}__{dataset_name}__{dataset_variant}__{module_name}__{run_name}'
        
        # set role for each subset
        selected_data_subsets_role = data_subsets_role[subset_collection_name].copy()
        
        # check if validation dataset is available, or it is 
        if isinstance(selected_data_subsets_role["valid"], float):
            train_proportion_applied = 1-selected_data_subsets_role["valid"] 
            selected_data_subsets_role["valid"] = None
        else:
            train_proportion_applied = train_proportion # used, only if there will be None for valid role, 
        
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
                df_subset = df_subset.sort_values(sort_by, ascending=False)
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
        results_list, model_predictions_dict, model_parameter_list, model_history_dict = CNN_GridSearch(
                # input data
                method                = method_name, 
                grid                  = cycle_grid,
                file_namepath_table   = subset_collection_composition_df,            

                # names to safe, used to identify input data & results
                dataset_name          = dataset_name,
                dataset_variant       = dataset_variant,
                module_name           = module_name,     
                run_name              = run_name,            

                # names used to search for subset names and save results
                class_encoding        = class_encoding,
                class_decoding        = class_decoding,
                dropout_value         = dropout_value, 
                train_subset_name     = selected_data_subsets_role["train"],                    # because I donth have to call that train in my files, 
                valid_subset_name     = selected_data_subsets_role["valid"],                    # if None, train_proportion will be used
                test_subset_name_list = selected_data_subsets_role["test"],         # must correspond to subset_name in file_namepath_table if None, the loist is simply shorter, 
                unit_test             = unit_test,
                train_proportion      = train_proportion_applied, # not available in that version, I would like to make it possible next version

                # ... results and info, 
                store_predictions=True,
                track_progres=display_partial_results,   
                plot_history=plot_history, # applied only if verbose==True
                model_fit__verbose=0,
                verbose=False            
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


            
      
      
      
      

# Function, ........................................................
def train_dense_network_NN_models_iteratively(*, 
    # names 
    run_ID="run01",
    dataset_name,
    dataset_variant,
    module_name,

    # define input data
    subset_composition_list, 
    data_subsets_role,
    subset_collection_names,

    # model parameters
    method_name ="knn", # {knn, svm, logreg, random_forest}
    grid,
    # model training cycle parameters,
    models_selected_at_each_cycle=0.3,                # int, or float, if float, a top franction is used, if int, a top nr of models is used, 
    include_method_variant_with_model_selection=True, # bool, if True, top models_selected_at_each_cycle wiht different model variant will be selected to next cycle
    include_random_nr_with_model_selection=False,     # bool, if True, top models_selected_at_each_cycle wiht different random nr will be selected to next cycle
    sort_models_by = "model_acc_valid",           # str {"model_acc_valid", "model_acc_train", "model_acc_test"}, in the future I will add other metrics 

    # saving
    save_path,
    save_partial_results=True, 
                                      
    # other,                         
    class_encoding,
    class_decoding,
    valid_proportion=0.2, # float, 0-1, on how much of the data shdould be randomly sorted into train subset, used only if valid datasusbet role is == None,  
    dropout_value=None, 
    unit_test=False,
    plot_history=True,
    verbose=False                                    
):                               

    # to make it easier later on,
    input_data_variant = {
        "dataset_name": dataset_name,
        "dataset_variant":dataset_variant,
        "module_name":module_name
        }

    # find dataframes with names and path to files to load
    subset_collection_composition_dict = prepare_list_with_subset_collection_composition_list( 
                subset_collection_names = subset_collection_names,       # names of subsets collection varinats, that have the same module, and dataset name and variant but different composition of batches,
                subset_collection_composition = subset_composition_list, # list with dict, where one of values is df wiht subset composition 
                **input_data_variant 
        )

    # train the model in cycles using different subset collections and parameters,
    train_and_test_cnn_tranfer_learning_models(  

        # names, 
        run_ID=run_ID,        # str, Unique ID added to name of each file to allow running and saving similar module create with different parameters or different data
        **input_data_variant,

        # define input data
        subset_collection_names            = subset_collection_names,           # list, anything that will allow you to identify which subset and run you are using             
        subset_collection_composition_dict = subset_collection_composition_dict, # list, wiht dataframes, the same lenght as subset_collection_names, returned by prepare_list_with_subset_collection_composition_list
        data_subsets_role                  = data_subsets_role,                 # dict, with names of subsets used as train, valid and test + in case you wish to use

        # model parameters, 
        method_name = method_name, # str, 
        grid        = grid,        # ParameterGrid object

        # model selection cycles,                       
        models_selected_at_each_cycle = models_selected_at_each_cycle,  # int, how many models with best performace will be selected and trained with another round of training and with next subset collection 
        include_method_variant_with_selection = include_method_variant_with_model_selection, # bool, if True, top models_selected_at_each_cycle wiht different model variant will be selected to next cycle
        include_random_nr_with_selection = include_random_nr_with_model_selection,     # bool, if True, top models_selected_at_each_cycle wiht different random nr will be selected to next cycle
        sort_models_by   = sort_models_by,           # str {"model_acc_valid"} 

        # pipe variables, # same for all cycles, 
        save_path        = save_path,             # str, eg PATH_results from configs 
        save_partial_results=save_partial_results, 
      
        
        class_encoding   = class_encoding,        # dict, global key: class_name, value:int 
        class_decoding   = class_decoding,        # dict, global key: int, value:class_name
        train_proportion = 1-valid_proportion,    # used only if validation datasets is not specified, 
        dropout_value    = dropout_value,         # str, from configs, 
        unit_test        = unit_test,

        # other, 
        plot_history=plot_history,
        verbose=verbose  
    )   


        



