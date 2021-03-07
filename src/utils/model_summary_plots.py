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
from src.utils.data_loaders import load_encoded_imgbatch_using_logfile, load_raw_img_batch
from src.utils.tools_for_plots import create_class_colors_dict
from src.utils.example_plots_after_clustering import plot_img_examples, create_spaces_between_img_clusters, plot_img_examples_from_dendrogram
from src.utils.annotated_pie_charts import annotated_pie_chart_with_class_and_group, prepare_img_classname_and_groupname





# Function, ........................................................................................................
# new
def plot_grid_acc_and_return_summary_df(*, 
    data, 
    module_names, 
    create_figure=True,
    replace_none = "default",                        
    # ...
    fig_title_prefix="", 
    fig_dname        = "kernel"  ,     
    subplots_dname   = "pca_components_used",
    xaxis_dname      = "gamma",
    linecolor_dname  = "C",
    linestyle_dnames = ["train", "valid"],
    # ...
    x_label = "SVM, gamma value",
    y_label = "Accuracy",
    y_limits = (0, 1)                            
    ):
    """
        Custom function created for plotting accuracy results obrained with SVM apporach 
        with model trained on the same data, using different parameters, C, gamma, and PCA and kernel
        with or wirhout PCA used in data preprocessing step, where 
        
        Caution, this func tion require full grid search of all paramteres, otherwise I suggest to use seasborn catplots, 
        
        # inputs
        . .............................................................................................
        . data.               : pd. DataFrame, created by random_forest_grid_search()
        . fig_title_prefix    : str, will be added to figure title, 
        . create_figure       : bool, if False, the function will retiunr only summary table, 
        . module_names        : list, with names given to different moduless or methods used 
                                for feature extraction from images, 
        . replace_none        : str, or None (default), what valud to use to replcace NaN or None values in input data, 
                                often used as parameters in piepeline  that may otherwise be incorrectly interpretted, 
                                while making a plot, 
        # ....
        . fig_dname           : str, variable name used for creating different figures, laying on top of each other, eg: "kernel"       
        . subplots_dname      : str, variable name used for creating different subplots, eg: "pca_components_used"
        . xaxis_dname         : str, variable name used for x-axis values, eg: "gamma"
        . linecolor_dname     : str, variable name used for creating different linecolors,
        . linestyle_dnames    : list with strings by default: ["train", "valid"]
                                each str, variable name used for creating lines with different style,  eg :, and --
        # ....
        . x_label             : str, eg: = "SVM, gamma value"
        . y_label             : str,  eg:  "Accuracy"
        . y_limits            : tuple, with int's eg: (0,1)
                       
        # returns,
        . .............................................................................................
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
        # DATA PREPARATION

        # extract data for one module, 
        df_subset = df_full.loc[df_full.module==module_name,:]

        # remove None values - not used here
        
        """
            here is an example of code that could be adapted to that, 
            in case it will be neede with other screened parameters, 
            
            
        """ 
        if replace_none!=None:
            for i in range(df_subset.shape[1]):
                new_value = "default"
                idx_to_replace = np.where(df_subset.iloc[:,i].isnull()==True)[0]
                if idx_to_replace.shape[0]>0:
                    df_subset.iloc[list(idx_to_replace),i] = new_value
                else:
                    pass
        else:
            pass
      

        # get lists of relevant features to plot, ..............
        
        # legacy code
        data_types = linestyle_dnames
        
        # .. for different, figsure and subplots, 
        fig_dnames = (df_subset.loc[:,fig_dname]).unique() # for different figures,
        subplots_dnames = (df_subset.loc[:,subplots_dname]).unique()  # for different subplots, 

        # .. on each subplots,
        xaxis_dnames = df_subset.loc[:, xaxis_dname].unique() # on x-axis
        linecolor_dnames = df_subset.loc[:, linecolor_dname].unique() # different line colors, 
        # set available color lines and styles
        available_linecolors = (["violet", "grey", "forestgreen", "blue", "red", "black", "orange", "cyan"]*40)[0:len(linecolor_dnames.tolist())] # for different tree max depth, 
        availabel_linestyles = (["-", "--", ":"]*40)[0:len(data_types)]

        # set positions of each point on x axis
        x_axis_possitions = np.arange(xaxis_dnames.shape[0])
        x_axis_labels = xaxis_dnames.astype(str)

        
        

        # .......................................................
        # FIGURES,

        # create separate fiugures for each dtype source speciffied on the top (they will be in paralel to make them easy to compare)
        for one_fig_dname in fig_dnames:

            # create figure
            ncols = subplots_dnames.shape[0]
            nrows = 1 # to allow comparing similar conditions in different columns, 
            figsize= (ncols*6, nrows*5)
            # ...
            if create_figure==True:
                fig, axs = plt.subplots(ncols=ncols, nrows=nrows, facecolor="white", figsize=figsize)
                fig.suptitle(f"{fig_title_prefix}{module_name}, {one_fig_dname}", fontsize=30)


            # subset data create with one pca procedure sheme
            for one_subplot_dname_i in range(subplots_dnames.shape[0]):

                # create subplot_description
                subplot_description.append(subplot_description[-1]+1) # fot summary table
                subplot_title_prefix = f"Plot {subplot_description[-1]+1}"

                if create_figure==True:            
                    ax = axs.flat[one_subplot_dname_i]

                max_acc_on_subplot  = 0 # initiating new search, 
                data_for_one_subplot = df_subset.loc[(df_subset.loc[:, subplots_dname]==subplots_dnames[one_subplot_dname_i]) & (df_subset.loc[:, fig_dname]==one_fig_dname),:]

                # now select colors and line types, 

                # subset series of data created with different max_depth   == different color
                for linecolor_dname_i, one_linecolor_dname in enumerate(linecolor_dnames):

                    # subeset data created with one data_type  == different line type
                    for data_type_i, data_type in enumerate(data_types):

                        # selet the data, 
                        data_for_one_line_on_one_subplot = data_for_one_subplot.loc[data_for_one_subplot.loc[:, linecolor_dname]==one_linecolor_dname,:]
                        data_for_one_line_on_one_subplot = data_for_one_line_on_one_subplot.loc[:,list(data_for_one_line_on_one_subplot.columns==f"model_acc_{data_type}")]
                        # print(data_for_one_line_on_one_subplot.shape, data_for_one_line_on_one_subplot, f"model_acc_{data_type}")            

                        # select color line and linetype
                        linecolor = available_linecolors[linecolor_dname_i]
                        linestyle = availabel_linestyles[data_type_i]

                        # add line to plot, 
                        if create_figure==True:
                            ax.plot(x_axis_possitions, data_for_one_line_on_one_subplot.values, color=linecolor, ls=linestyle, label=f"{data_type}, {linecolor_dname}: {one_linecolor_dname}")
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
                    if subplots_dnames[one_subplot_dname_i]==0:  
                        sb_title=f"No PCA"
                    else: 
                        sb_title=f"{subplot_title_prefix}. PCA, {subplots_dnames[one_subplot_dname_i]} components"
                        
                    ax.set_title(sb_title, fontsize=15)    
                    ax.set_xlabel(x_label, fontsize=15)
                    ax.set_ylabel(y_label, fontsize=15)
                    ax.set(ylim=y_limits)           

                    # .. grid,
                    ax.grid(color="grey", ls="--", lw=0.3)

                    # .. axes,
                    ax.spines["right"].set_visible(False) # and below, remove white border, 
                    ax.spines["top"].set_visible(False)

                    # .. ticks, 
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
                l = fig.legend(handles, labels, loc=(0.9, 0.1), fontsize=1, frameon=False) # bbox_to_anchor = (0,-0.1,1,1)
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
    columns_to_present_all = ["Plot_nr", "model_ID", "method", "module", 
                              "model_acc_train", "model_acc_valid","model_acc_test", 
                              fig_dname, subplots_dname, xaxis_dname, linecolor_dname,
                              "baseline_acc_train", "baseline_acc_valid", "baseline_acc_test"]
    df_temp = df_temp.loc[:, columns_to_present_all]

    return df_temp

  

  
  
  
  
# Function, .........................................................................................

def visual_model_summary(*, 
    model_predictions_dict, 
    model_ID=0,
    n=100, 
    examples_to_plot="all",
    dataset_name,                              
    subset_name,
    img_batch_subset_names, 
    path_to_raw_img_batch,
    class_colors,
    make_plot_with_img_examples=True,
    plot_classyfication_summary=True, 
    max_img_per_col=10,
    add_proba_values_to_img_name=True, 
    pie_charts_in_ne_row = 6,
    title_prefix="",
    first_pie_title="Classyfication Results",
    second_pie_title="True Class Assigment",
    pie_data_for_all_images_in_img_batch=True,
    use_new_colors_for_predicted_classes=False, # also new, decides whther to add purple or class colors to legend and plot opn pie plot 2
    #### added later on pie, for description see annotated pie chart function,                     
    PIE_legend_loc = "upper right",
    PIE_ax_title_fonsize_scale=1,
    PIE_legend_fontsize_scale=1,
    PIE_wedges_fontsize_scale=1,
    PIE_legend_ncol=4,
    PIE_tight_lyout=True,
    PIE_title_ha="right",
    PIE_figsze_scale=1,  
    PIE_subplots_adjust_top=0.9,
    PIE_ax_title_fontcolor=None,
    ###  
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
                               
        . pie_data_for_all_images_in_img_batch : bool, if True, pie charts will display summary based on classyfication results in all
                              img batch loaded, not only eymple images plotted,  
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
    
        IMPORTNAT COMMENT
        - typically, I package the results by module, that was used to create the, 
          howeevr, here, I was also creating a summary tbale that allows easily to compare, 
          the models created with different modules, and all models created by all modules are package into one dictionary, 
          with different model_ID that corresponds to model_ID in summary table. 
          
    
    
    """    
    
    
    # two little legacy issuess, 
    module_nr = model_ID # small legacy name, to avoid revriting some parts, 
    run=True # a bit of legacy again, sorry for that, 
    if run==True: 
        
        
        # small correction to parameters to avoid mabiguous results on plot later on, 
        if isinstance(n, str): # ie "all" 
            examples_to_plot = "all"
        else:
            pass
        
        # extract info required for plots, 
        original_labels = model_predictions_dict[model_ID][subset_name[0]]["original_labels"]
        model_predictions = model_predictions_dict[model_ID][subset_name[0]]["model_predictions"]
        acc_restuls_and_params = model_predictions_dict[model_ID][subset_name[0]]["acc_restuls_and_params"]
        original_img_idx_in_batch = model_predictions_dict[model_ID][subset_name[0]]["idx_in_batch"] # created because I was using train_test_split
        
        if add_proba_values_to_img_name==True:
            model_predictions_proba = model_predictions_dict[model_ID][subset_name[0]]["model_predictions_proba"]
            class_decoding = model_predictions_dict[model_ID][subset_name[0]]['class_decoding']
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
                    row= np.array(row).flatten().astype(float)
                    #class_idx = np.where(row==row.max())[0][0]
                    img_name = f"{np.round(row.max()*100,1)}%"
                    return img_name
                
                # ..
                img_names = pd.DataFrame(model_predictions_proba).apply(
                    create_image_description, 
                    sorted_class_names=np.array(list(class_decoding.values())), 
                    axis=1
                )
                
                # .. finally, add predicted classes to img_name
                
                
                
                
            else:
                img_names = pd.Series([f"image {x}" for x in original_img_idx_in_batch.tolist()])
            img_names = img_names.values[img_idx].tolist()  
    
        
            # disable some fucntiosn in the plot, when only small nr of images is displayed - to make it nice looking, 
            if len(img_idx)>1:
                # fig with img examples, 
                subplots_adjust_top=0.75
                title = f"{title_prefix} ({len(img_idx)} items from {raw_img_batch.shape[0]} in the batch), classification: {examples_to_plot} results"
                class_colors_for_legend = class_colors
                
            else:
                title = None
                pie_title = None
                class_colors_for_legend = None

                
            # create img_names and img_groupnames
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
        
        # create annotated pie charts title, 
        if pie_data_for_all_images_in_img_batch==False and np.array(searched_predictions).sum()>0:
            pie_title_1 = f"{first_pie_title} ({len(img_idx)} items from {raw_img_batch.shape[0]} in batch), {examples_to_plot} results"    
            pie_title_2 = f"{second_pie_title} ({len(img_idx)} items from {raw_img_batch.shape[0]} in batch),{examples_to_plot} results"    
        if pie_data_for_all_images_in_img_batch==True:
            pie_title_1 = f"{first_pie_title} ({raw_img_batch.shape[0]} items from {raw_img_batch.shape[0]} in batch), All results"
            pie_title_2 = f"{second_pie_title} ({raw_img_batch.shape[0]} items from {raw_img_batch.shape[0]} in batch), All results"
        else:
            pie_title_1=None
            pie_title_2=None
        
        
        # here I implamented LAZY solution with a bit of spagettiti code, it was however easier at that time :P, sorry
        if plot_classyfication_summary==True:
            if pie_data_for_all_images_in_img_batch==False:
                if np.array(searched_predictions).sum()>0:
                    
                    # pie charts 1. showing composition of classes clreated with the model
                    annotated_pie_chart_with_class_and_group(
                            title=pie_title_1,
                            classnames=np.array(original_labels)[img_idx].tolist(), 
                            class_colors=class_colors,
                            ###
                            groupnames=np.array([f"Classified as:\n{x}" for x in model_predictions.tolist()])[img_idx].tolist(), 
                            #groupname_colors=class_colors,
                            mid_pie_circle_color='grey', # instead of the above,
                            ###
                            n_subplots_in_row=pie_charts_in_ne_row, 
                            ###
                            legend_loc=PIE_legend_loc,
                            legend_fontsize_scale=PIE_legend_fontsize_scale,
                            wedges_fontsize_scale=PIE_wedges_fontsize_scale,
                            legend_ncol=PIE_legend_ncol,
                            tight_lyout=PIE_tight_lyout, # Bool
                            ax_title_fonsize_scale=PIE_ax_title_fonsize_scale,
                            title_ha=PIE_title_ha,
                            figsze_scale=PIE_figsze_scale,
                            subplots_adjust_top=PIE_subplots_adjust_top,
                            ax_title_fontcolor=PIE_ax_title_fontcolor
                            )
                    
                    # pie charts 2. showing how each True class was classiifed 
                    
                    if use_new_colors_for_predicted_classes==True:
                        # .. create distionary with clusternames and assigned colors, 
                        class_color_for_clusters = create_class_colors_dict(
                            list_of_unique_names = pd.Series(np.array([f"Classified as: \n{x}" for x in model_predictions.tolist()])[img_idx].tolist()).unique().tolist(),
                            cmap_name="Purples", cmap_colors_from = 0.2, cmap_colors_to = 1
                                )
                    else:
                        # instead of createting new colors for predicted classes, you will re-map colors form true classes onto new names of predicted classes
                        class_color_for_clusters = dict(zip([f"Classified as: \n{x}" for x in list(class_colors.keys())], list(class_colors.values())))


                    
                    # ...
                    annotated_pie_chart_with_class_and_group(
                            title=pie_title_2,
                            classnames=np.array([f"Classified as: \n{x}" for x in model_predictions.tolist()])[img_idx].tolist(),
                            class_colors=class_color_for_clusters,
                            ###
                            groupnames=np.array(original_labels)[img_idx].tolist(),  
                            groupname_colors=class_colors, 
                            ###
                            n_subplots_in_row=pie_charts_in_ne_row,
                            ###
                            legend_loc=PIE_legend_loc,
                            legend_fontsize_scale=PIE_legend_fontsize_scale,
                            wedges_fontsize_scale=PIE_wedges_fontsize_scale,
                            legend_ncol=PIE_legend_ncol,
                            tight_lyout=PIE_tight_lyout, # Bool
                            ax_title_fonsize_scale=PIE_ax_title_fonsize_scale,
                            title_ha=PIE_title_ha,
                            figsze_scale=PIE_figsze_scale,
                            subplots_adjust_top=PIE_subplots_adjust_top,
                             ax_title_fontcolor=PIE_ax_title_fontcolor
                            )
                    
                else:
                    pass
                    
                    
            else:
                annotated_pie_chart_with_class_and_group(
                        title=pie_title_1,
                        classnames=np.array(original_labels).tolist(), 
                        class_colors=class_colors,
                        ###
                        groupnames=np.array([f"Classified as: \n{x}" for x in model_predictions.tolist()]).tolist(), 
                        #groupname_colors=class_colors, 
                        ###
                        n_subplots_in_row=pie_charts_in_ne_row, 
                        ###
                        legend_loc=PIE_legend_loc,
                        legend_fontsize_scale=PIE_legend_fontsize_scale,
                        wedges_fontsize_scale=PIE_wedges_fontsize_scale,
                        legend_ncol=PIE_legend_ncol,
                        tight_lyout=PIE_tight_lyout, # Bool
                        ax_title_fonsize_scale=PIE_ax_title_fonsize_scale,
                        title_ha=PIE_title_ha,
                        figsze_scale=PIE_figsze_scale,
                        subplots_adjust_top=PIE_subplots_adjust_top,
                        ax_title_fontcolor=PIE_ax_title_fontcolor # if none, automatic colors will be added
                        )    
                
                # pie charts 2. showing how each True class was classiifed 
                    
                if use_new_colors_for_predicted_classes==True:
                    # .. create distionary with clusternames and assigned colors, 
                    class_color_for_clusters = create_class_colors_dict(
                            list_of_unique_names = pd.Series(np.array([f"Classified as: \n{x}" for x in model_predictions.tolist()]).tolist()).unique().tolist(),
                            cmap_name="Purples", cmap_colors_from = 0.2, cmap_colors_to = 1
                                    )   
                else:
                    # instead of createting new colors for predicted classes, you will re-map colors form true classes onto new names of predicted classes
                    class_color_for_clusters = dict(zip([f"Classified as: \n{x}" for x in list(class_colors.keys())], list(class_colors.values())))

                # ...
                annotated_pie_chart_with_class_and_group(
                            title=pie_title_2,
                            classnames=np.array([f"Classified as: \n{x}" for x in model_predictions.tolist()]).tolist(),
                            class_colors=class_color_for_clusters,
                            ###
                            groupnames=np.array(original_labels).tolist(),  
                            groupname_colors=class_colors, 
                            ###
                            n_subplots_in_row=pie_charts_in_ne_row, 
                            ###
                            legend_loc=PIE_legend_loc,
                            legend_fontsize_scale=PIE_legend_fontsize_scale,
                            wedges_fontsize_scale=PIE_wedges_fontsize_scale,
                            legend_ncol=PIE_legend_ncol,
                            tight_lyout=PIE_tight_lyout, # Bool
                            ax_title_fonsize_scale=PIE_ax_title_fonsize_scale,
                            title_ha=PIE_title_ha,
                            figsze_scale=PIE_figsze_scale,
                            subplots_adjust_top=PIE_subplots_adjust_top,
                             ax_title_fontcolor=PIE_ax_title_fontcolor
                            )  
        else:
            pass
     
    
    
    
    
    
    

    
    
# Function, ........................................................................................................

def model_gridsearch_summary_plots(*, 
    data, 
    module_names, 
    create_figure=True,
    replace_none = "default",                        
    # ...
    fig_title_prefix="", 
    fig_dname        = ""  ,     
    subplots_dname   = "pca_components_used",
    xaxis_dname      = "",
    linecolor_dname  = "",
    linestyle_dnames = ["train", "valid"],
    # ...
    figsize = (12,4),
    fontscale = 0.6, 
    x_label = None,
    y_label = "Accuracy",
    y_limits = (0, 1),
    subplots_adjust_top=0.7
    ):
    """
        Custom function created for plotting accuracy results obrained with SVM apporach, and then generalized,  
        to create accuracy of different models trained on the same data, using different parameters, eg: C, gamma, and PCA and kernel
        Models trained with withnout PCA or with different number of PCA composents, are on different subplots, It shoudl also work on 
        all other vriabvles as

        Caution, this func tion require full grid search of all paramteres, otherwise I suggest to use seasborn catplots, 
        
        # inputs
        . .............................................................................................
        . data.               : pd. DataFrame, created by random_forest_grid_search()
        . fig_title_prefix    : str, will be added to figure title, 
        . create_figure       : bool, if False, the function will retiunr only summary table, 
        . module_names        : list, with names given to different moduless or methods used 
                                for feature extraction from images, 
        . replace_none        : str, or None (default), what valud to use to replcace NaN or None values in input data, 
                                often used as parameters in piepeline  that may otherwise be incorrectly interpretted, 
                                while making a plot, 
        # ....
        . fig_dname           : str, variable name used for creating different figures, laying on top of each other, eg: "kernel"       
        . subplots_dname      : str, variable name used for creating different subplots, eg: "pca_components_used"
        . xaxis_dname         : str, variable name used for x-axis values, eg: "gamma"
        . linecolor_dname     : str, variable name used for creating different linecolors,
        . linestyle_dnames    : list with strings by default: ["train", "valid"]
                                each str, variable name used for creating lines with different style,  eg :, and --
        # ....
        . figsize             : tuple, with two int, 
        . x_label             : str, eg: = "SVM, gamma value"
        . y_label             : str,  eg:  "Accuracy"
        . y_limits            : tuple, with int's eg: (0,1)
        . subplots_adjust_top : float from 0 to 1, corresponding to value of fig.subplots_adjust(top=?)
        . fontscale           : float from >0 to any value, on how much ti scale preseted fontscales on the plot
                
        # returns,
        . .............................................................................................
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
        # DATA PREPARATION

        # extract data for one module, 
        df_subset = df_full.loc[df_full.module==module_name,:]

        # remove None values - not used here
        
        """
            here is an example of code that could be adapted to that, 
            in case it will be neede with other screened parameters, 
        """ 
        if replace_none!=None:
            for i in range(df_subset.shape[1]):
                new_value = "default"
                idx_to_replace = np.where(df_subset.iloc[:,i].isnull()==True)[0]
                if idx_to_replace.shape[0]>0:
                    df_subset.iloc[list(idx_to_replace),i] = new_value
                else:
                    pass
        else:
            pass
      

        # get lists of relevant features to plot, ..............
        
        # legacy code
        data_types = linestyle_dnames
        
        # .. for different, figsure and subplots, 
        fig_dnames = (df_subset.loc[:,fig_dname]).unique() # for different figures,
        subplots_dnames = (df_subset.loc[:,subplots_dname]).unique()  # for different subplots, 

        # .. on each subplots,
        xaxis_dnames = df_subset.loc[:, xaxis_dname].unique() # on x-axis
        linecolor_dnames = df_subset.loc[:, linecolor_dname].unique() # different line colors, 
        # set available color lines and styles
        available_linecolors = (["violet", "grey", "forestgreen", "blue", "red", "black", "orange", "cyan"]*40)[0:len(linecolor_dnames.tolist())] # for different tree max depth, 
        availabel_linestyles = (["-", "--", ":"]*40)[0:len(data_types)]

        # set positions of each point on x axis
        x_axis_possitions = np.arange(xaxis_dnames.shape[0])
        x_axis_labels = xaxis_dnames.astype(str)

        
        

        # .......................................................
        # FIGURES,

        # create separate fiugures for each dtype source speciffied on the top (they will be in paralel to make them easy to compare)
        subplot_nr = 0 #. ust be done here, because I am suing one additional subplot to have nice legend, 
        for one_fig_dname in fig_dnames:

            # create figure
            ncols = subplots_dnames.shape[0]+1
            nrows = 1 # to allow comparing similar conditions in different columns, 
            if create_figure==True:
                fig, axs = plt.subplots(ncols=ncols, nrows=nrows, facecolor="white", figsize=figsize)
                fig.suptitle(f"{fig_title_prefix} {one_fig_dname}", fontsize=30*fontscale)
            
            
            # ..
            for one_subplot_dname_i in range(ncols):

                if one_subplot_dname_i<(ncols-1):
                    subplot_nr +=1

                    # subset data create with one pca procedure sheme
                    leg_linestyle_list = list()
                    leg_linecolor_list = list()
                    leg_label_list = list()

                    # create subplot_description
                    subplot_description.append(subplot_nr) # fot summary table
                    subplot_title_prefix = f"Plot {subplot_nr}"

                    if create_figure==True:            
                        ax = axs.flat[one_subplot_dname_i]

                    max_acc_on_subplot  = 0 # initiating new search, 
                    data_for_one_subplot = df_subset.loc[(df_subset.loc[:, subplots_dname]==subplots_dnames[one_subplot_dname_i]) & (df_subset.loc[:, fig_dname]==one_fig_dname),:]

                    # now select colors and line types, 

                    # subset series of data created with different max_depth   == different color
                    for linecolor_dname_i, one_linecolor_dname in enumerate(linecolor_dnames):

                        # subeset data created with one data_type  == different line type
                        for data_type_i, data_type in enumerate(data_types):

                            # selet the data, 
                            data_for_one_line_on_one_subplot = data_for_one_subplot.loc[data_for_one_subplot.loc[:, linecolor_dname]==one_linecolor_dname,:]
                            data_for_one_line_on_one_subplot = data_for_one_line_on_one_subplot.loc[:,list(data_for_one_line_on_one_subplot.columns==f"model_acc_{data_type}")]
                            # print(data_for_one_line_on_one_subplot.shape, data_for_one_line_on_one_subplot, f"model_acc_{data_type}")            

                            # select color line and linetype
                            linecolor = available_linecolors[linecolor_dname_i]
                            linestyle = availabel_linestyles[data_type_i]

                            # add line to plot, 
                            if create_figure==True:
                                ax.plot(x_axis_possitions, data_for_one_line_on_one_subplot.values, color=linecolor, ls=linestyle, label=f"{data_type}, {linecolor_dname}: {one_linecolor_dname}")
                                leg_linestyle_list.append(linestyle)
                                leg_linecolor_list.append(linecolor)
                                leg_label_list.append(f"{data_type}, {linecolor_dname}: {one_linecolor_dname}")
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

                    # this is in caxye I woudl like to have one legend on the fugure, 
                    if create_figure==True:
                        
                        
                        # set up subplot title,
                        if subplots_dname=="pca_components_used":
                            if subplots_dnames[one_subplot_dname_i]==0:  
                                sb_title=f"{subplot_title_prefix}, No PCA"
                            else: 
                                sb_title=f"{subplot_title_prefix}. PCA, {subplots_dnames[one_subplot_dname_i]} components"
                        if subplots_dname!="pca_components_used":
                            sb_title=f"{subplot_title_prefix}. {subplots_dname} = {subplots_dnames[one_subplot_dname_i]}"

                        # check for x label, 
                        if x_label==None:
                            x_label=xaxis_dname
                        else:
                            x_label=x_label
                            
                        # .. titles 
                        ax.set_title(sb_title, fontsize=15*fontscale)    
                        ax.set_xlabel(x_label, fontsize=15*fontscale)
                        ax.set_ylabel(y_label, fontsize=15*fontscale)
                        ax.set(ylim=y_limits)           

                        # .. grid,
                        ax.grid(color="grey", ls="--", lw=0.3)

                        # .. axes,
                        ax.spines["right"].set_visible(False) # and below, remove white border, 
                        ax.spines["top"].set_visible(False)

                        # .. ticks, 
                        ax.set_xticks(x_axis_possitions)
                        ax.set_xticklabels(x_axis_labels, fontsize=10*fontscale, rotation=-70)
                        ax.tick_params(axis='both', labelsize=(10*fontscale))

                        # legend, 
                        handles, labels = ax.get_legend_handles_labels()

                # ..      
                if one_subplot_dname_i==(ncols-1):
                    if create_figure==True:
                        ax = axs.flat[one_subplot_dname_i]               
                        ax.spines["left"].set_visible(False) # and below, remove white border, 
                        ax.spines["bottom"].set_visible(False)
                        ax.spines["right"].set_visible(False) # and below, remove white border, 
                        ax.spines["top"].set_visible(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # create legend using labels and handles form the last subplot 
                        leg = ax.legend(handles, labels, loc="center", fontsize=10*fontscale, frameon=False)
                        leg.set_title('Legend',prop={'size':(15*fontscale)}) # otherwise, I can not control fontsize of the title    
    
    
            # adjust subplots margins,  
            if create_figure==True:
                fig.tight_layout()
                fig.subplots_adjust(top=subplots_adjust_top)
                plt.show();  
            else:
                pass

    # create summary table to return with best model (using validation accuracy, in each data subsset displayed on each subplot, )
    df_temp = df_full.iloc[df_fullidx_from_best_performing_models_with_each_parameter,: ]
    df_temp.reset_index(drop=True, inplace=True)

    # add column with corresponding subplot_title_prefix
    df_temp["Plot_nr"]=pd.Series([f"Plot {x}" for x in subplot_description[1::]])

    # reorder
    columns_to_present_all = ["Plot_nr", "model_ID", "method", "module", 
                              "model_acc_train", "model_acc_valid","model_acc_test", 
                              fig_dname, subplots_dname, xaxis_dname, linecolor_dname,
                              "baseline_acc_train", "baseline_acc_valid", "baseline_acc_test"]
    df_temp = df_temp.loc[:, columns_to_present_all]

    return df_temp

  