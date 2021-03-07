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

from src.utils.image_augmentation import * # to create batch_labels files, 
from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch
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





# Function, ...........................................................................................................
def random_forest_grid_search(*, 
      method_name="random_forest", 
      path, 
      dataset_name, 
      subset_names_tr, 
      subset_names_te, 
      module_names, 
      class_encoding, 
      grid, 
      param_names_for_DecisionTreeClassifier, 
      train_proportion=0.7, 
      random_state_nr=0, 
      store_predictions=True, 
      track_progres=False, 
      verbose=False):

    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function,         Custom function that perfomes grid search using decision trees, on features extracted 
                            from images with different tf.hub modules. 
                            
                            
                            Optionally, it allows using pca, for tranforming 
                            extracted features intro selected number of principial components, later on used by decision 
                            tree algorithm
                            
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
          none, 

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

        if track_progres==True:
            print(f"{i} {module_name} _________________________________________ {pd.to_datetime('now')}")
        else:
            pass
        
        
        # Grid search, 
        for params in grid:
            
            
            if track_progres==True:
                print('.', end="")
            else:
                pass
            
            
            # PARAMETERS, ...................................
            model_ID +=1
            pca_axes_nr = params["pca"]
            dt_params_dct = dict(zip(param_names_for_DecisionTreeClassifier,[params[x] for x in param_names_for_DecisionTreeClassifier]))
            # ...
            Xy_names = ["train", "valid", "test"]


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
                print(f'{model_ID}: {module_name}, logfie: {logfiles[0]}')
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

            
            # ...................
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
            # Create decision tree classifier,  

            # Create decission tree
            #dt = RandomForestClassifier(random_state=random_state_nr) # this could be modiffied if needed, 

            # Create decission tree & define parameteres, 
            dt = RandomForestClassifier(random_state=random_state_nr, **dt_params_dct)
            dt.fit(X_dct["train"], y_dct["train"])
            
            # .. get accuracy,
            model_acc = dict()
            for xyname in Xy_names:
                model_acc[f"model_acc_{xyname}"] = dt.score(X_dct[xyname], y_dct[xyname])

            if verbose==True:
                print(" --- ", model_ID, model_acc)    
            else:
                pass
            

            # ..................
            # Export decision tree for graphviz
            
            # first Extract single tree
            one_tree = dt.estimators_[0]
            # more info on that issue:
            # https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
            
            # then use standard notation, 
            dot_data = export_graphviz(
                one_tree,                               # INPUT; decision tree obj, 
                out_file=None,                    # OUTPUT; if None, the obj, is returned as GraphViz dot format (string)

                # feature, class names to display, 
                # feature_names=features.columns,        # name of the features in X
                class_names=list(class_encoding.keys()), # names of the features in Y
                max_depth=dt_params_dct['max_depth'],    # optional, If None, the tree is fully generated.

                # starts displayed in each node,
                node_ids=False,                   # shows node ID number, 
                impurity=True,                    # True, is default, eg shows gini in each node,  
                proportion=True,                  # display proportion of each class, in each node
                precision=2,                      # number of digits displayed for each float,

                # tree orientation, 
                rotate=False,                     # False; the three in top-down orientation 

                # node aestetics,
                filled=True,                      # add color, to nodes, 
                rounded=True,                     # round corners in node boxes, 

                # leave features,
                leaves_parallel=False              # display all leaves at the bottom
            )

            # COLLECT THE RESULTS ,..............................  
            'acc_restuls_and_params were added to all objects in case I woudl have some dounbts about results origine,'

            # collect acc_restuls_and_params
            acc_restuls_and_params = {
                 "model_ID": model_ID,
                 "method": method_name,
                 "module": module_name,
                 **baseline_acc,
                 **model_acc,
                 **dt_params_dct,
                 "pca_components_used":pca_axes_nr,
            }
            model_acc_and_parameters_list.append(acc_restuls_and_params) # in list, so it can be used as pd.df immediately, 
            
            # collect trees, 
            dot_data_dict[int(model_ID)] = {
                "dot_data": dot_data, 
                "acc_restuls_and_params": acc_restuls_and_params
            }


            
            # Collect Model predictions, 
            if store_predictions==True:
                one_model_predictions = dict()
                for xyname in Xy_names:
                    # make predictions and decode them,
                    predictions         = dt.predict(X_dct[xyname])
                    decoded_predictions = pd.Series(predictions).map(class_decoding).values
                    model_predictions_proba = dt.predict_proba(X_dct[xyname])
                    decoded_y_labels    = pd.Series(y_dct[xyname]).map(class_decoding).values
                    # ...
                    one_model_predictions[xyname] = {
                        "idx_in_batch":       idx_y_dct[xyname],
                        "original_labels":         decoded_y_labels, 
                        "model_predictions":       decoded_predictions, 
                        "model_predictions_proba": model_predictions_proba,
                        "acc_restuls_and_params":  acc_restuls_and_params,
                        "class_decoding": class_decoding
                    }# added, in case I woudl have some dounbts about results origine, 

                # and finally, add this to the big dict wiht all the results, 
                model_predictions_dict[model_ID] = one_model_predictions
            
            else:
                model_predictions_dict[model_ID] = None

        if track_progres==True:
            print(f"\nDONE _________________________________________ {pd.to_datetime('now')}",end="\n\n")
        else:
            pass            
            
    
    
    # ..................................................
    return model_acc_and_parameters_list, dot_data_dict, model_predictions_dict
            
    
    
    
    
    
    
    
    
    
    
    


# Function, ........................................................................................................

def plot_random_tree_summary_andor_table(*, data, module_names, fig_title_prefix="", create_figure=True):
    """
        Custom function created for plotting results obrained with random forest apporach 
        with model trained on the dsame data, using different parameters, max_depth, n_estimators, class_weight
        with or wirhout PCA used in data preprocessing step, where 
        
        # inputs
        . data.               : pd. DataFrame, created by random_forest_grid_search()
        . fig_title_prefix    : str, will be added to figure title, 
        . create_figure       : bool, if False, the function will retiunr only summary table, 
        . module_names        : list, with names given to different moduless or methods used 
                                for feature extraction from images, 
        # returns, 
        . Plot                : matplotlib figure, with n submplots eqial tzo number of different conditions 
                                for data preprocessing steps, 
        . summary table.      : pd. DataFrame,  with data on best performing model on each subplot, 
                                models are selected based on validation dataset accuracy, 
        
        # Notes
        . Caution ,function may return warning, 
                   I havent tracked it because the values are correct in the table (tested on several datasets)
                   and becauxse I had to deliver this function ASAP, 
    
    """
    
    # because I wanted to have nicer name and I preffer to work on copy
    df_full = data.copy()

    # list to store some results, 
    df_fullidx_from_best_performing_models_with_each_parameter = list()
    subplot_description = [0]# created to allow faster navigation thrue summary table in comparison to subplots, 

    for module_name in module_names:

        # .......................................................
        # DATA TYPES PRESENTED ON PLOTS,   
        data_types = ["train", "valid"] # data types


        # .......................................................
        # DATA PREPARATION

        # extract data for one module, 
        df_subset = df_full.loc[df_full.module==module_name,:]

        # remove None values
        for i in range(df_subset.shape[1]):
            if df_subset.columns[i]=="class_weight": new_value = "uniform"
            if df_subset.columns[i]=="max_depth": new_value = "full"   
            # ..
            idx_to_replace = np.where(df_subset.iloc[:,i].isnull()==True)[0]
            if idx_to_replace.shape[0]>0:
                df_subset.iloc[list(idx_to_replace),i] = new_value
            else:
                pass

        # get lists of relevant features to plot, ..............
        # .. for different, figsure and subplots, 
        class_weight_params = df_subset.class_weight.unique() # for different figures,
        pca_components_used = df_subset.pca_components_used.unique() # for different subplots, 

        # .. on each subplots,
        n_estimators_params = df_subset.n_estimators.unique() # on x-axis
        max_depth_params = df_subset.max_depth.unique() # different line colors, 
        # data_types =  different line types, 

        # set available color lines and styles
        available_linecolors = (["violet", "grey", "forestgreen", "blue", "red", "black", "orange", "cyan"]*40)[0:len(max_depth_params.tolist())] # for different tree max depth, 
        availabel_linestyles = (["-", "--", ":"]*40)[0:len(data_types)]

        # set positions of each point on x axis
        x_axis_possitions = np.arange(n_estimators_params.shape[0])
        x_axis_labels = n_estimators_params.astype(int)


        # .......................................................
        # FIGURES,

        # create separate fiugure for each class weight (they will be in paralel to make them easy to compare)
        for class_w in class_weight_params:

            # create figure
            ncols = pca_components_used.shape[0]
            nrows = 1 # to allow comparing similar conditions in different columns, 
            figsize= (ncols*6, nrows*5)
            # ...
            if create_figure==True:
                fig, axs = plt.subplots(ncols=ncols, nrows=nrows, facecolor="white", figsize=figsize)
                fig.suptitle(f"{fig_title_prefix}{module_name}, class weights: {class_w}", fontsize=30)


            # subset data create with one pca procedure sheme
            for subplot_i in range(ncols):

                # create subplot_description
                subplot_description.append(subplot_description[-1]+1) # fot summary table
                subplot_title_prefix = f"Plot {subplot_description[-1]+1}"

                if create_figure==True:            
                    ax = axs.flat[subplot_i]

                max_acc_on_subplot  = 0 # initiating new search, 
                data_for_one_subplot = df_subset.loc[(df_subset.pca_components_used==pca_components_used[subplot_i]) & (df_subset.class_weight==class_w),:]

                # now select colors and line types, 

                # subset series of data created with different max_depth   == different color
                for max_depth_i, one_max_depth in enumerate(max_depth_params):

                    # subeset data created with one data_type  == different line type
                    for data_type_i, data_type in enumerate(data_types):

                        # selet the data, 
                        data_for_one_line_on_one_subplot = data_for_one_subplot.loc[data_for_one_subplot.max_depth==one_max_depth,:]
                        data_for_one_line_on_one_subplot = data_for_one_line_on_one_subplot.loc[:,list(data_for_one_line_on_one_subplot.columns==f"model_acc_{data_type}")]
                        # print(data_for_one_line_on_one_subplot.shape, data_for_one_line_on_one_subplot, f"model_acc_{data_type}")            

                        # select color line and linetype
                        linecolor = available_linecolors[max_depth_i]
                        linestyle = availabel_linestyles[data_type_i]

                        # add line to plot, 
                        if create_figure==True:
                            ax.plot(x_axis_possitions, data_for_one_line_on_one_subplot.values, color=linecolor, ls=linestyle, label=f"{data_type}, max deph: {one_max_depth}")
                        else:
                            pass

                        # using the occasion collect df_fullidx_from_best_performing_models_with_each_parameter
                        if data_type == "valid":
                            if max_acc_on_subplot < data_for_one_line_on_one_subplot.max().values[0]:
                                idx_max_postion  = np.where(data_for_one_line_on_one_subplot.values.flatten()==data_for_one_line_on_one_subplot.max().values[0])[0][0]
                                idx_max = list(data_for_one_line_on_one_subplot.index)[idx_max_postion]
                                max_acc_on_subplot = data_for_one_line_on_one_subplot.max().values[0]

                # add row with max value into the list, at the end of each subplot,     
                df_fullidx_from_best_performing_models_with_each_parameter.append(idx_max)


                # add aestetist to subplot, 
                if create_figure==True:
                    # .. ax, title and axes descrition, and limits,
                    if pca_components_used[subplot_i]==0:sb_title=f"No PCA"
                    else: sb_title=f"{subplot_title_prefix}. PCA, {pca_components_used[subplot_i]} components"
                    ax.set_title(sb_title, fontsize=15)    
                    ax.set_xlabel("tree number (n_estimators)", fontsize=15)
                    ax.set_ylabel("Accuracy", fontsize=15)
                    ax.set(ylim=(0,1.05))           

                    # .. grid,
                    ax.grid(color="grey", ls="--", lw=0.3)

                    # .. axes,
                    ax.spines["right"].set_visible(False) # and below, remove white border, 
                    ax.spines["top"].set_visible(False)

                    # .. ticks, 
                    ax.set_yticks([0, 0.2, .4, .6, .8, 1])
                    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=12)
                    ax.set_xticks(x_axis_possitions)
                    ax.set_xticklabels(x_axis_labels, fontsize=12)

                    # legend, 
                    handles, labels = ax.get_legend_handles_labels()
                    #l = ax.legend(handles, labels, loc=(0.6, 0.005), fontsize=10, frameon=False) # bbox_to_anchor = (0,-0.1,1,1)
                    # l.set_title('Data Types',prop={'size':10}) # otherwise, I can not control fontsize of the title,
                else:
                    pass

            # this is in caxye I woudl like to have one legend on the fugure, 
            if create_figure==True:
                l = fig.legend(handles, labels, loc=(0.90, 0.1), fontsize=12, frameon=False) # bbox_to_anchor = (0,-0.1,1,1)
                l.set_title('Legend',prop={'size':12}) # otherwise, I can not control fontsize of the title            

                # adjust subplots margins,  
                fig.tight_layout()
                fig.subplots_adjust(top=0.70, right=0.9)
                plt.show();  
            else:
                pass

    # create summary table to return with best model (using validation accuracy, in each data subsset displayed on each subplot, )
    df_temp = df_full.iloc[df_fullidx_from_best_performing_models_with_each_parameter,: ]
    df_temp.reset_index(drop=True, inplace=True)

    # add column with corresponding subplot_title_prefix
    df_temp["Plot_nr"]=pd.Series([f"Plot {x}" for x in subplot_description[1::]])

    # reorder
    columns_to_present_all = ["Plot_nr", "model_ID", "method", "module", "model_acc_train", "model_acc_valid",
                              "model_acc_test", "class_weight", "pca_components_used", "max_depth", 
                              "n_estimators", "baseline_acc_train", "baseline_acc_valid", "baseline_acc_test"]
    df_temp = df_temp.loc[:, columns_to_present_all]

    return df_temp



