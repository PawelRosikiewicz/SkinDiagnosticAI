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
import matplotlib.pyplot as plt # for making plots, 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

from src.utils.image_augmentation import * # to create batch_labels files, 
from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch
from src.utils.example_plots_after_clustering import plot_img_examples, create_spaces_between_img_clusters, plot_img_examples_from_dendrogram
from src.utils.annotated_pie_charts import annotated_pie_chart_with_class_and_group, prepare_img_classname_and_groupname



# Function, ...............................................................................

def my_logredCV(*,
    path,
    dataset_name,
    subset_names_tr,
    subset_names_te,
    module_names,
    class_encoding,
    pipe,
    grid_kwargs,
    method_name = "logreg_with_cv",
    cv_nr = 5,
    store_predictions =True,             
    verbose =False
):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function,         Custom function that perfomes grid search with cross validation 
                            using log-regression, on features extracted from images with different tf.hub modules. 
                            - the function returns, dataframe with accuracy in train, test and validation datsets
                              (that is also called as train datasets by the orignal function)
                              
                            Caution, unlike similar function created for grid search with decision treesm and pca, 
                            this funcitons, saves only predictions made by the best model, according to selection mede with
                            the gridcv fucntion, 
                            additionally, acc_restuls_and_params_df are dataframe, not as prevoiusly a ist with dictionaries, 
                            
        # Inputs,     
          .................................................................................................
        . path               : str, path to directory, with data sstored, 
        . dataset_name       : str, datassets name, used while creating all the data     
        . subset_names_tr    : list, eg: [train", "valid"], these two dastasets will be concastenated
        . subset_names_te    : list, eg: ["test"], these two dastasets will be concastenated in that order
                               Caution, I assumed that, more then one subset of data is keept in dataset file folder, ¨
                               eg, that you stored test and train data separately, 
        . module_names       : list, with names given to different moduless or methods used for feature extractio 
                               from images, 
        . pipe               : Pipeline from sklearn.pipeline
        . grid_kwargs        : dict, with parameters for GridSearchCV from sklearn.model_selection
        . cv_nr              : how many train/tests subset to create with crosvalidation, 
        . store_predictions  : bool, if True, predictions for all models, with train, validations and test datasets 
                               will be perfomed and stored in  model_predictions_dict        
        . class_encoding     : dict, key:<orriginal class name>:value<numerical value used by decision tre>
                               eg: dict(zip(list(class_colors.keys()), list(range(len(class_colors)))))

        # Returns,     
          .................................................................................................
          
          
          . model_acc_and_parameters_list : list, where each entry is a dict, with accuracy and parameters usied to build 
                                    a given model, and model_ID that can be used to retrieve items from two other 
                                    objectes returned by this funciotn, 
          
          . model_predictions_dict : dict, key=model_ID ()
                                     content: another dict, with "train, test and valid" keys 
                                            representing predictions made with eahc of these subsets
                                     each of them is also a dict. with, 
                                      >  "idx_in_used_batch":       index of each image in original img_batch 
                                                                    as created, using, subset_names_tr, and 
                                                                    load_encoded_imgbatch_using_logfile() function
                                      >  "original_labels":         array, with original class names for each image in a given dataset
                                      >  "model_predictions":       array, with preducted class names for each image in a given dataset
                                      >. "model_predictions_proba"  array, probabilities for each class generated with logistic regression
                                      >  "acc_restuls_and_params":  contains dict, with acc_restuls_and_params         
                                                                    to ensure reproducibility, 
                                      > "class_decoding"            dict, with int, representing column nr in model_predictions_proba, 
                                                                    and class name 
        # Notes,     
          .................................................................................................                                                                
            Caution you may got the follwioign message:
            UserWarning: A worker stopped while some jobs were given to the executor. This can be caused 
            by a too short worker timeout or by a memory leak. "timeout or by a memory leak.", UserWarning
            no worries, 
            - it often happnes, when cv grid is using all available processors on your computer, 
            - it shodul not alter the results, 


    """

    # objects to store results, and class decoding, 
    model_predictions_dict = dict()
    class_decoding = dict(zip(list(list(class_encoding.values())), list(class_encoding.keys()))) # 


    for module_nr, module_name in enumerate(module_names):

        # ................................................................
        # DATA PREPARATION,

        # dataset names used  for results and calling data subsets, 
        '''Caution: this function uses cross-validation, 
           thus train and valid fdatatasets are comnbined'''
        
        # create Xy_names to inform on type of img_batches_that were used 
        Xy_names = ["train", "test"]

        # find any logfile created while saving img files, 
        os.chdir(path)
        logfiles = []
        for file in glob.glob(f"{''.join([module_name,'_',dataset_name])}*_logfile.csv"):
            logfiles.append(file)

        # ... info, 
        if verbose==True:
            print(f'{"".join(["."]*80)}')
            print(f'{module_nr}: {module_name}, logfie: {logfiles[0]}')
            print(f" --- Number of tested combinations {len(ParameterGrid(grid_kwargs))}")
        else:
            pass

        # load train data, 
        X, batch_labels = load_encoded_imgbatch_using_logfile(logfile_name=logfiles[0], load_datasetnames=subset_names_tr)
        X = X.astype(np.float)
        y = pd.Series(batch_labels.classname).map(class_encoding).values.astype("int")
        idx_y_tr = np.arange(y.shape[0], dtype="int")

        # and becuadse I am using cv instead of tr and valid test:
        X_tr = X
        y_tr = y

        # test data, 
        X_te, batch_labels = load_encoded_imgbatch_using_logfile(logfile_name=logfiles[0], load_datasetnames=subset_names_te)
        X_te = X_te.astype(np.float)
        y_te = pd.Series(batch_labels.classname).map(class_encoding).values.astype("int")
        idx_y_te = np.arange(y_te.shape[0], dtype="int")

        # place all X and y in dict so its easier later on, 
        X_dct = dict(zip(Xy_names, [X_tr, X_te]))
        y_dct = dict(zip(Xy_names, [y_tr, y_te]))
        idx_y_dct = dict(zip(Xy_names, [idx_y_tr, idx_y_te]))



        # ................................................................
        # TRAIN ESTIMATORS,

        # baseline, 

        # ... Create Most frequet baseline, 
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_dct[Xy_names[0]].astype(np.float), y_dct[Xy_names[0]].astype(int))
        baseline_acc = dict()
        #        scores are callated and added later on, 


        # grid search with cross validation, 

        # .. cv grid search using pipe, and grid_kwargs, 
        grid_cv = GridSearchCV(
            pipe,                    # pipeline, 
            grid_kwargs,             # grid, provided as dictionary with list or tupples with param values to test,
            cv=cv_nr,                # defaults k-fold, stratified strategy, 
            return_train_score=True,
            n_jobs=-1                # use all avaialable cores, 
            #verbose=1,              # to be inform on the progress !
        )

        # .. Fit estimator
        grid_cv.fit(X_dct[Xy_names[0]], y_dct[Xy_names[0]]);



        # ................................................................    
        # COLLECT THE RESULTS,
        'acc_restuls_and_params were added to all objects in case I woudl have some dounbts about results origine,'

        # collect acc_restuls_and_params

        #  extract and select relevant logreg_cv_results[model_ID]#
        columns = ["param_logreg__C", 'param_scaler', "mean_train_score", "std_train_score", "mean_test_score", "std_test_score"]
        one_module_acc_restuls_and_params_df =  pd.DataFrame(grid_cv.cv_results_).loc[:, columns]

        # add standard items as in the other methods, 
        one_module_acc_restuls_and_params_df["method"] =method_name
        one_module_acc_restuls_and_params_df["module"] =module_name

        # add baseline accuracy, 
        for xyname in Xy_names:
            one_module_acc_restuls_and_params_df[f"baseline_acc_{xyname}"] = dummy.score(X_dct[xyname], y_dct[xyname])

        # add model_ID
        if module_nr==0:
            model_IDs = list(range(one_module_acc_restuls_and_params_df.shape[0]))
            one_module_acc_restuls_and_params_df["model_ID"] = model_IDs
            # ..
            acc_restuls_and_params_df = one_module_acc_restuls_and_params_df
        else:
            id_from = acc_restuls_and_params_df.model_ID.iloc[-1]+1
            id_to = id_from+one_module_acc_restuls_and_params_df.shape[0]
            one_module_acc_restuls_and_params_df["model_ID"] = list(range(id_from, id_to))
            # ..
            acc_restuls_and_params_df = pd.concat([acc_restuls_and_params_df, one_module_acc_restuls_and_params_df], axis=0)

        # Collect Model predictions,  
        if store_predictions==True:
            best_model_predictions = dict()
            for xyname in Xy_names:
                # make predictions and decode them,
                predictions         = grid_cv.best_estimator_.predict(X_dct[xyname])
                decoded_predictions = pd.Series(predictions).map(class_decoding).values
                decoded_y_labels    = pd.Series(y_dct[xyname]).map(class_decoding).values
                # ...
                grid_cv.best_estimator_.predict_proba(X_dct[xyname])
                # ...
                save_name = "__".join(xyname)
                
                
                best_model_predictions[xyname] = {
                    "idx_in_batch":            idx_y_dct[xyname],
                    "original_labels":         decoded_y_labels, 
                    "model_predictions":       decoded_predictions, 
                    "model_predictions_proba": grid_cv.best_estimator_.predict_proba(X_dct[xyname]),
                    "acc_restuls_and_params":  grid_cv.best_estimator_,
                    "class_decoding": class_decoding
                }# added, in case I woudl have some dounbts about results origine, 

            # and finally, add this to the big dict wiht all the results, 
            model_predictions_dict[module_nr] = best_model_predictions           
        else:
            model_predictions_dict[module_nr] = None

            
    # prepare dataframe,        
    acc_restuls_and_params_df = acc_restuls_and_params_df.reset_index(drop=True)
    
    # ... change some column names, so they are the same as in other df's with the same results in my dataset, 
    acc_restuls_and_params_df  = acc_restuls_and_params_df.rename(columns={
      "mean_train_score":"model_acc_train",
      "std_train_score":"model_acc_train_std",
      "mean_test_score":"model_acc_valid",
      "std_test_score":"model_acc_valid_std",
    })

    # add accuracy for test dataset, with best model for which I created the predictions, 

    # ... add new column to df,
    acc_restuls_and_params_df["model_acc_test"]=np.nan
    
    # .. calulate test accuracy with best models with each module, 
    for module_nr, module_name in enumerate(module_names):
        # model_acc_train
        y_pred = model_predictions_dict[module_nr]["test"]["model_predictions"]
        y_true = model_predictions_dict[module_nr]["test"]["original_labels"]
        model_acc_test = accuracy_score(y_true, y_pred)

        # find wchich is the best performing, 
        idx_best_model = acc_restuls_and_params_df.loc[acc_restuls_and_params_df.module==module_name,:].sort_values("model_acc_valid", ascending=False).index.tolist()[0]

        # and place it there, 
        acc_test_columns_idx = np.where(acc_restuls_and_params_df.columns == "model_acc_test")[0][0] # I did that to avoid warnings, 
        acc_restuls_and_params_df.iloc[idx_best_model,acc_test_columns_idx]=model_acc_test        

    return  acc_restuls_and_params_df, model_predictions_dict
  
  
  
  
  
  

  
  
  
  
# Function, .........................................................................................

def plot_examples_with_predictions_and_proba(*, 
    model_predictions_dict, 
    n=100, 
    examples_to_plot="all",
    module_names, 
    dataset_name,                              
    subset_name,
    img_batch_subset_names, 
    path_to_raw_img_batch,
    class_colors,
    make_plot_with_img_examples=True,
    plot_classyfication_summary=True, 
    max_img_per_col=10,
    verbose=False         
                                             
):
    
    """
      Wrapper fuction that created 2 figures summarizing the results of image classyficaiton with logistic regression
      with image exmaples and summary of class assigment to classes created by that 
      each image has probability calulated with logreg. The functions can display n requested image exmaples with 
      their calssyfication or only correct or incorrectly classified images, if avaialable
      
      # Input
        -----------------------------------------------------------------------------------
        
        
        
        . model_predictions_dict: dict, created by my_logredCV() function, 
        . n.                 : int, or str {"all"},  how many examples are requested, to plot, 
                               if you use any str, value, eg all, all images and predictions will be used to created plots, 
        . examples_to_plot   : str, type of img, examples that can be plotted, {"all", "correct", "incorrect"}
                               based on classyfication results, 
                               CAUTION, the function will plot n examples, or max available examples if smaller number 
                               is in the given batch,If no examples are found, it will print the informations, if verbose==True, 
        . class_colors       : dictionary, {str <"class_name">: str <"color">}
                               CAUTION: colors and class names must be unique !
        . max_img_per_col    : max nr of images, that will be dissplayed in each column in eqch geroup, 
        . ...
        . path_to_raw_img_batch : str, path to directory, with data sstored, 
        . dataset_name       : str, datassets name, used while creating all the data     
        . subset_name        : list, eg: [train", "valid"], these are the names of datasubset in model_predictions_dict
                               They may be different, because I used grid search with cross validation, 
                               amd in the effect i had only train and test datsets, despite using more batches with different names
                               for each of them, ONLY ONE, is required, ie len(list)==1  
        . img_batch_subset_names: list, eg: [train", "valid"], these two dastasets will be concastenated
                               and used to load road_img_batch,                                            
        . module_names       : list, with names given to different moduless or methods used for feature extraction
                               from images,
      # Returns,
        -----------------------------------------------------------------------------------
      . 2 Figures,           : with image examples, using plot_img_examples() from src.utils.example_plots_after_clustering
                               and with pie charts using annotated_pie_chart_with_class_and_group() from src.utils.annotated_pie_charts

      # Notes:
        -----------------------------------------------------------------------------------
      . None, 
    
    """    
    
    for module_nr, module_name in enumerate(module_names):    

        # extract info required for plots, 
        original_labels = model_predictions_dict[module_nr][subset_name[0]]["original_labels"]
        model_predictions = model_predictions_dict[module_nr][subset_name[0]]["model_predictions"]
        model_predictions_proba = model_predictions_dict[module_nr][subset_name[0]]["model_predictions_proba"]
        acc_restuls_and_params = model_predictions_dict[module_nr][subset_name[0]]["acc_restuls_and_params"]
        class_decoding = model_predictions_dict[module_nr][subset_name[0]]['class_decoding']


        # Load raw img,  
        "reloading each time to avoid having problems"
        raw_img_batch = load_raw_img_batch(
                                    load_datasetnames=img_batch_subset_names, 
                                    path=path_to_raw_img_batch, 
                                    image_size=(500,500), verbose=False)


        # select images for plot, 

        if examples_to_plot=="incorrect":
            searched_predictions = [x!=y for x, y in  zip(original_labels.tolist(), model_predictions.tolist())]
            sel_img_idx = np.arange(0, raw_img_batch.shape[0])[searched_predictions]

        if examples_to_plot=="correct":
            searched_predictions = [x==y for x, y in  zip(original_labels.tolist(), model_predictions.tolist())]
            sel_img_idx = np.arange(0, raw_img_batch.shape[0])[searched_predictions]

        if examples_to_plot=="all":
            searched_predictions = [True]*raw_img_batch.shape[0]
            sel_img_idx = np.arange(0, raw_img_batch.shape[0])[searched_predictions]

        # if there are no examples, to display, here is an option to stop
        if np.array(searched_predictions).sum()==0:
            if verbose==True:
                print(f"No - {examples_to_plot} - image example found in that dataset")
            else:
                pass

        if np.array(searched_predictions).sum()>0:

            # check whther selection is required at all, 
            if isinstance(n, int):
                # create up to n examples, where possible, using sel_img_idx
                which_idx_to_use = np.unique(np.floor(np.linspace(0,sel_img_idx.shape[0], n, endpoint=False)).astype(int)).tolist()
                img_idx = sel_img_idx[which_idx_to_use] 
                
            if isinstance(n, str):
                # use asll instances, and all images - 
                #.  t is especially designed to work with plot_classyfication_summary==True, and lot_img_examples==False
                img_idx = np.arange(0, raw_img_batch.shape[0])                
                
                
                
                
            # create img names, with class name and probability 

            # .. helper funciton, 
            def create_image_description(row, sorted_class_names):
                row= np.array(row).flatten()
                class_idx = np.where(row==row.max())[0][0]
                img_name = f"{sorted_class_names[class_idx]}: {np.round(row[class_idx]*100,1)}%"
                return img_name
            # ..
            img_names = pd.DataFrame(model_predictions_proba).apply(
                create_image_description, 
                sorted_class_names=np.array(list(class_decoding.values())), 
                axis=1
            )

            # disable some fucntiosn in the plot, when only small nr of images is displayed - to make it nice looking, 
            if len(img_idx)>1:
                # fig with img examples, 
                subplots_adjust_top=0.75
                title = f"{module_name}, results: {examples_to_plot} ({len(img_idx)} available examples from {raw_img_batch.shape[0]} in total)"
                class_colors_for_legend = class_colors
                pie_title = None
                
            else:
                title = None
                pie_title = None
                class_colors_for_legend = None

                
            # create img_names and img_groupnames
            if examples_to_plot=="incorrect":
                img_groupname = ["Inorectly Classified Images"]*len(img_idx)
            else:
                img_groupname = np.array([f"Classified as:\n   {x}" for x in model_predictions.tolist()])[img_idx].tolist()
            img_names = img_names.values[img_idx].tolist()
            
            
            # plot image examples
            if make_plot_with_img_examples==True:
                plot_img_examples(
                    selected_img_batch        = raw_img_batch[img_idx],
                    img_groupname             = img_groupname,
                    img_name                  = img_names,
                    img_color                 = pd.Series(original_labels).map(class_colors).values[img_idx].tolist(),
                    class_colors_for_legend   = class_colors_for_legend,
                    title                     = title,
                    legend_loc                = "center",
                    max_img_per_col           = max_img_per_col,
                    figsize_scaling           = 3,
                    space_between_clusters    = 0.5,
                    subplots_adjust_top       = subplots_adjust_top, 
                    space_for_color_box_factor= 0.01,
                    fontScale                 = 2,
                    img_name_fontcolor        = "lime"
                )
            else:
                pass

            # Plot Pie charts summarizing items classified into each class, 
            if plot_classyfication_summary==True:
                annotated_pie_chart_with_class_and_group(
                    title=pie_title,
                    classnames=np.array(original_labels)[img_idx].tolist(), 
                    class_colors=class_colors,
                    ###
                    groupnames=np.array([f"Classified as:\n{x}" for x in model_predictions.tolist()])[img_idx].tolist(), 
                    #groupname_colors=class_colors, 
                    ###
                    n_subplots_in_row=6, 
                    legend_loc="upper right"
                    )
            else:
                pass


              
              
              
              
# Function, .........................................................................................

def plot_examples_with_predictions_and_proba_gamma(*, 
    model_predictions_dict, 
    model_ID=0,
    n=100, 
    examples_to_plot="all",
    module_name="", 
    dataset_name,                              
    subset_name,
    img_batch_subset_names, 
    path_to_raw_img_batch,
    class_colors,
    make_plot_with_img_examples=True,
    plot_classyfication_summary=True, 
    max_img_per_col=10,
    add_proba_values_to_img_name=True, 
    verbose=False                                          
):
    
    """
      Wrapper fuction that created 2 figures summarizing the results of image classyficaiton with logistic regression
      with image exmaples and summary of class assigment to classes created by that 
      each image has probability calulated with logreg. The functions can display n requested image exmaples with 
      their calssyfication or only correct or incorrectly classified images, if avaialable
      
      Cazution: this is function, beta adapted to plot summary results for other tetchiques without probability of classyficaiton in name 
      it can be used with almost any technique that had model_ID ir module as model name in model_predictions_dict
      
      version beta was depreciated with version gamma, 
      
      # Input
        -----------------------------------------------------------------------------------
        
        # new in beta
        . add_proba_values_to_img_name   : bool, if True, it works as model_predictions_dict()
        . model_ID.                      : str or int, key in model_predictions_dict for a given model,
        
        . ....
        . model_predictions_dict: dict, created by my_logredCV() function, 
        . n.                 : int, or str {"all"},  how many examples are requested, to plot, 
                               if you use any str, value, eg all, all images and predictions will be used to created plots, 
        . examples_to_plot   : str, type of img, examples that can be plotted, {"all", "correct", "incorrect"}
                               based on classyfication results, 
                               CAUTION, the function will plot n examples, or max available examples if smaller number 
                               is in the given batch,If no examples are found, it will print the informations, if verbose==True, 
        . class_colors       : dictionary, {str <"class_name">: str <"color">}
                               CAUTION: colors and class names must be unique !
        . max_img_per_col    : max nr of images, that will be dissplayed in each column in eqch geroup, 
        . ...
        . path_to_raw_img_batch : str, path to directory, with data sstored, 
        . dataset_name       : str, datassets name, used while creating all the data     
        . subset_name        : list, eg: [train", "valid"], these are the names of datasubset in model_predictions_dict
                               They may be different, because I used grid search with cross validation, 
                               amd in the effect i had only train and test datsets, despite using more batches with different names
                               for each of them, ONLY ONE, is required, ie len(list)==1  
        . img_batch_subset_names: list, eg: [train", "valid"], these two dastasets will be concastenated
                               and used to load road_img_batch,                                            
        . module_names       : list, with names given to different moduless or methods used for feature extraction
                               from images,
      # Returns,
        -----------------------------------------------------------------------------------
      . 2 Figures,           : with image examples, using plot_img_examples() from src.utils.example_plots_after_clustering
                               and with pie charts using annotated_pie_chart_with_class_and_group() from src.utils.annotated_pie_charts

      # Notes:
        -----------------------------------------------------------------------------------
      . None, 
        Img order on plots,  : images are ordered as in model_predictions_dict as selected and shuflled by the 
                               test_train split, each group is treated independently,
                               groups are ordered according to their size, from the largest one to 
                               the smallest one, or by order in which the first picture appeared in batch labels, 
                               - this version. was selected as default, 
    """    
    
    
    
    module_nr = model_ID # small legacy name, to avoid revriting some parts, 
    
    run=True
    if run==True:   

        # extract info required for plots, 
        original_labels = model_predictions_dict[module_nr][subset_name[0]]["original_labels"]
        model_predictions = model_predictions_dict[module_nr][subset_name[0]]["model_predictions"]
        acc_restuls_and_params = model_predictions_dict[module_nr][subset_name[0]]["acc_restuls_and_params"]
        original_img_idx_in_batch = model_predictions_dict[module_nr][subset_name[0]]["idx_in_batch"] # created because I was using train_test_split
        
        if add_proba_values_to_img_name==True:
            model_predictions_proba = model_predictions_dict[module_nr][subset_name[0]]["model_predictions_proba"]
            class_decoding = model_predictions_dict[module_nr][subset_name[0]]['class_decoding']
        else:
            pass
        
        
        
        

        # Load raw img,  
        "reloading each time to avoid having problems"
        raw_img_batch = load_raw_img_batch(
                                    load_datasetnames=img_batch_subset_names, 
                                    path=path_to_raw_img_batch, 
                                    image_size=(500,500), verbose=False)

        
        # .. select and reorder images that were created uwing test_train_split functions, 
        raw_img_batch = raw_img_batch[original_img_idx_in_batch.tolist(),:]

        # select images for plot, 

        if examples_to_plot=="incorrect":
            searched_predictions = [x!=y for x, y in  zip(original_labels.tolist(), model_predictions.tolist())]
            sel_img_idx = np.arange(0, raw_img_batch.shape[0])[searched_predictions]

        if examples_to_plot=="correct":
            searched_predictions = [x==y for x, y in  zip(original_labels.tolist(), model_predictions.tolist())]
            sel_img_idx = np.arange(0, raw_img_batch.shape[0])[searched_predictions]

        if examples_to_plot=="all":
            searched_predictions = [True]*raw_img_batch.shape[0]
            sel_img_idx = np.arange(0, raw_img_batch.shape[0])[searched_predictions]

        # if there are no examples, to display, here is an option to stop
        if np.array(searched_predictions).sum()==0:
            if verbose==True:
                print(f"No - {examples_to_plot} - image example found in that dataset")
            else:
                pass

        if np.array(searched_predictions).sum()>0:

            # check whther selection is required at all, 
            if isinstance(n, int):
                # create up to n examples, where possible, using sel_img_idx
                which_idx_to_use = np.unique(np.floor(np.linspace(0,sel_img_idx.shape[0], n, endpoint=False)).astype(int)).tolist()
                img_idx = sel_img_idx[which_idx_to_use] 
                
            if isinstance(n, str):
                # use asll instances, and all images - 
                #.  t is especially designed to work with plot_classyfication_summary==True, and lot_img_examples==False
                img_idx = np.arange(0, raw_img_batch.shape[0])                
                
            # create img names, with class name and probability 
            if add_proba_values_to_img_name==True:
                # .. helper funciton, 
                def create_image_description(row, sorted_class_names):
                    row= np.array(row).flatten()
                    class_idx = np.where(row==row.max())[0][0]
                    img_name = f"{sorted_class_names[class_idx]}: {np.round(row[class_idx]*100,1)}%"
                    return img_name
                # ..
                img_names = pd.DataFrame(model_predictions_proba).apply(
                    create_image_description, 
                    sorted_class_names=np.array(list(class_decoding.values())), 
                    axis=1
                )
            else:
                img_names = pd.Series([f"image {x}" for x in original_img_idx_in_batch.tolist()])
            img_names = img_names.values[img_idx].tolist()  
        

            # disable some fucntiosn in the plot, when only small nr of images is displayed - to make it nice looking, 
            if len(img_idx)>1:
                # fig with img examples, 
                subplots_adjust_top=0.75
                title = f"{module_name}, results: {examples_to_plot} ({len(img_idx)} available examples from {raw_img_batch.shape[0]} in total)"
                class_colors_for_legend = class_colors
                pie_title = None
                
            else:
                title = None
                pie_title = None
                class_colors_for_legend = None

                
            # create img_names and img_groupnames
            if examples_to_plot=="incorrect":
                img_groupname = ["Inorectly Classified Images"]*len(img_idx)
            else:
                img_groupname = np.array([f"Classified as:\n   {x}" for x in model_predictions.tolist()])[img_idx].tolist()

            
            # plot image examples
            if make_plot_with_img_examples==True:
                plot_img_examples(
                    selected_img_batch        = raw_img_batch[img_idx],
                    img_groupname             = img_groupname,
                    img_name                  = img_names,
                    img_color                 = pd.Series(original_labels).map(class_colors).values[img_idx].tolist(),
                    class_colors_for_legend   = class_colors_for_legend,
                    title                     = title,
                    legend_loc                = "center",
                    max_img_per_col           = max_img_per_col,
                    figsize_scaling           = 3,
                    space_between_clusters    = 0.5,
                    subplots_adjust_top       = subplots_adjust_top, 
                    space_for_color_box_factor= 0.01,
                    fontScale                 = 2,
                    img_name_fontcolor        = "lime"
                )
            else:
                pass

            # Plot Pie charts summarizing items classified into each class, 
            if plot_classyfication_summary==True:
                annotated_pie_chart_with_class_and_group(
                    title=pie_title,
                    classnames=np.array(original_labels)[img_idx].tolist(), 
                    class_colors=class_colors,
                    ###
                    groupnames=np.array([f"Classified as:\n{x}" for x in model_predictions.tolist()])[img_idx].tolist(), 
                    #groupname_colors=class_colors, 
                    ###
                    n_subplots_in_row=6, 
                    legend_loc="upper right"
                    )
            else:
                pass
     