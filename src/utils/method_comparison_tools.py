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








# Function, ..............................................................................
def boxplot_with_acc_from_different_models(*, title="", summary_df, dtype, 
                                           figsize=(10,4), legend__bbox_to_anchor=(0.5, 1.3), 
                                           cmap="tab10", cmap_colors_from=0, cmap_colors_to=1, 
                                           legend_ncols=1):
    """
        Small helper function to crreate nice boxplot with the data 
        provided by summary df, after exploring many different models
        
        # inputs
        dtype       : str {"test", "train", "valid"}
        summary_df  : summary dataframe with accuracy and parameter 
                      results returned by grid search developed for all models 
        figsize.    : tuple, two integers, eg: (10, 5)
                      
        # returns, 
        boxplot     : 
        
        summary_df=summary_df_for_boxplot
        figsize=(10,4)
        dtype="valid"
        t = True
        if t==True:
        
    """
    
    # ...............................................
    # data preparation

    # find all modules
    module_names = summary_df.module.unique().tolist()

    # get results for each method, 
    bx_data = list() # list with arr's with values for each box,  
    bx_labels = list() # on x-axis, for each box, 
    bx_modules = list() # set as different colors, 
    # ..
    for module_name in module_names:
        one_module_summary_df = summary_df.loc[summary_df.module==module_name,:]
        # ..
        for one_method in one_module_summary_df.method.unique().tolist():

            acc_data = one_module_summary_df.loc[one_module_summary_df.method==one_method,f"model_acc_{dtype}"]
            acc_data = acc_data.dropna()

            if len(acc_data)>0:
                # ...
                bx_data.append(acc_data.values)
                bx_labels.append(one_method)
                bx_modules.append(module_name)
            else:
                pass

    # find memdians and reorder 
    bx_data_medians = list()
    for i, d in enumerate(bx_data):
        bx_data_medians.append(np.median(d))

    # ...
    temp_df = pd.DataFrame({
        "labels": bx_labels,
        "medians": bx_data_medians,
        "modules": bx_modules
    })
    new_order = temp_df.sort_values("medians", ascending=True).index.values
    # ...
    ordered_bx_data = list()
    ordered_bx_labels = list()
    ordered_bx_modules = list()
    for i in new_order:
        ordered_bx_data.append(bx_data[i])
        ordered_bx_labels.append(bx_labels[i])
        ordered_bx_modules.append(bx_modules[i])

    # ...............................................    
    # set colors for different modules,  
    module_colors = create_class_colors_dict(
        list_of_unique_names=module_names, 
        cmap_name=cmap, 
        cmap_colors_from=cmap_colors_from, 
        cmap_colors_to=cmap_colors_to
    )

    # ...............................................
    # boxplot,  - plt.boxplot(ordered_bx_data);
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.suptitle(title, fontsize=20)

    # add boxes,
    bx = ax.boxplot(ordered_bx_data, 
            showfliers=True,                  # remove outliers, because we are interested in a general trend,
            vert=True,                        # boxes are vertical
            labels=ordered_bx_labels,           # x-ticks labels
            patch_artist=True,
            widths=0.3
    )
    ax.grid(ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(ordered_bx_labels, rotation=45, fontsize=12, ha="right")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=15)
    ax.set_ylabel("Accuracy\n", fontsize=20)
    ax.set_xlabel("Method", fontsize=20)
    ax.set_ylim(0,1.02)

    # add colors to each box individually,
    
    for i, j in zip(range(len(bx['boxes'])),range(0, len(bx['caps']), 2)) :
        median_color  ="black"
        box_color     = module_colors[ordered_bx_modules[i]]

        # set properties of items with the same number as boxes,
        plt.setp(bx['boxes'][i], color=box_color, facecolor=median_color, linewidth=2, alpha=0.8)
        plt.setp(bx["medians"][i], color=median_color, linewidth=2)
        plt.setp(bx["fliers"][i], markeredgecolor="black", marker=".") # outliers

        # set properties of items with the 2x number of features as boxes,
        plt.setp(bx['caps'][j], color=median_color)
        plt.setp(bx['caps'][j+1], color=median_color)
        plt.setp(bx['whiskers'][j], color=median_color)
        plt.setp(bx['whiskers'][j+1], color=median_color)

        
    # set xtick labels,  
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(module_colors[ordered_bx_modules[i]])

    if len(module_names)>0:

        # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
        patch_list_for_legend =[]
        for i, m_name in enumerate(list(module_colors.keys())):
            label_text = f"{m_name}"
            patch_list_for_legend.append(mpl.patches.Patch(color=module_colors[m_name], label=label_text))

        # add patches to plot,
        fig.legend(
            handles=patch_list_for_legend, frameon=False, 
            scatterpoints=1, ncol=legend_ncols, 
            bbox_to_anchor=legend__bbox_to_anchor, fontsize=15)

        # create space for the legend
        fig.subplots_adjust(top=0.8)    

        
    # add line with baseline
    acc_baseline = summary_df.loc[:,f"baseline_acc_{dtype}"].dropna().values.flatten()
    ax.axhline(acc_baseline[0], lw=2, ls="--", color="dimgrey")
    ax.text(len(ordered_bx_data)+0.4, acc_baseline[0]+0.05, "most frequent baseline", ha="right", color="dimgrey", fontsize=15)        
        
        
    # color patches behing boxplots, 
    patch_width = 2
    patch_color = "lightgrey"
    pathces_starting_x = list(range(0, len(ordered_bx_data), patch_width*2))
    # ...
    for i, sx in enumerate(pathces_starting_x):
        rect = plt.Rectangle((sx+0.5, 0), patch_width, 1000, color=patch_color, alpha=0.8, edgecolor=None)
        ax.add_patch(rect)        
        
        
    # color patches for styling the accuracy, 
    rect = plt.Rectangle((0,0), len(ordered_bx_data)*100, acc_baseline[0], color="red", alpha=0.2, edgecolor=None)
    ax.add_patch(rect)          
    rect = plt.Rectangle((0,acc_baseline[0]), len(ordered_bx_data)*100, 0.7-acc_baseline[0], color="orange", alpha=0.2, edgecolor=None)
    ax.add_patch(rect)             
    rect = plt.Rectangle((0,0.7), len(ordered_bx_data)*100, 10, color="forestgreen", alpha=0.2, edgecolor=None)
    ax.add_patch(rect)            
        
        
        


    # ...............................
    return fig



    
    







# Function, ..............................................................................
def method_comparison_boxplot(*, 
    title="Accuracy of models created with each method\n",  
    data,         # pd.DataFrame with the results,   
    figsize=(10,4),
    # ...
    col_with_results,       # df colname with values to display, eg: test_accuracy ...
    col_with_group_names,   # df colname with values that will be displayed as names of each box (these do not have to be unique)
    col_with_group_ID,      # df colname with values that will be grouped for separate boxes (must be unieque)
    col_with_group_colors,  # df colname with values that will have different colors (colors can not be mixed within diffeent group_ID)
    # ... colors
    cmap="tab10",
    cmap_colors_from=0, 
    cmap_colors_to=1,                               
    # .. legend
    legend__bbox_to_anchor=(0.9, 1.15), 
    subplots_adjust_top = 0.8,
    legend_ncols=4,
    # .. baseline
    baseline_title="",       # "most frequent baseline",
    baseline_loc = -0.05,
    baseline = 0.25,          
    top_results = 0.9,      # green zone on a plot, 
    # ... fontsize
    title_fontsize=20,
    legend_fontsize=10,
    xticks_fontsize=10,
    yticks_fontsize=15,
    axes_labels_fontsize=20,
    # ... axies labels
    xaxis_label = "Method",
    yaxis_label = "Accuracy\n",
    paint_xticks=False
):
    """
        Nice function to create ngs-like boxplots for comparison of acc of differemnt model groups
        it is more generic version of the abofe function, 
    """

    
    # ...............................................
    # managment
    Stop_Function = False
    
    # ...............................................
    # data preparation - step.1 extraction
    # ...............................................

    # - extract unique values that will be searched, 
    unique_group_ID               = data.loc[:,col_with_group_ID].unique().tolist()
    unique_group_color_names      = data.loc[:,col_with_group_colors].unique().tolist()
    
    # - map colors onto color_groups_names
    bx_color_dict = create_class_colors_dict(
        list_of_unique_names=unique_group_color_names, 
        cmap_name=cmap, 
        cmap_colors_from=cmap_colors_from, 
        cmap_colors_to=cmap_colors_to
    )   
    
    # - lists with data for boxes, 
    'one item for one box in each'
    bx_data             = []  
    bx_names            = []
    bx_colors           = []
    bx_id               = []
    bx_colors_dict_key  = []
    
    # - separate all boxes, and then find out what is the color and data associated with that box
    for one_group_ID in unique_group_ID:
        bx_id.append(one_group_ID)
        
        # get the data and other columns for one box
        data_df_for_one_box = data.loc[data.loc[:, col_with_group_ID]==one_group_ID,:]
        
        # find out, data, name and color to display
        # .... num. data ....
        bx_data.append(data_df_for_one_box.loc[:,col_with_results].values) # np.array
        
        
        # .... labels .....
        one_bx_label = data_df_for_one_box.loc[:,col_with_group_names].unique().tolist()
        if len(one_bx_label)==1:
            bx_names.append(one_bx_label[0]) # np.array
        else:
            if verbose==1:
                print(f"{one_group_ID} contains more then one group to display wiht different names !")
            else:
                Stop_Function = True
                pass
  

        # .... colors ....
        one_box_color = data_df_for_one_box.loc[:,col_with_group_colors].map(bx_color_dict).iloc[0]
        color_test_values = data_df_for_one_box.loc[:,col_with_group_colors].unique().tolist()
        if len(color_test_values)==1:
            bx_colors.append(one_box_color) # np.array
            bx_colors_dict_key.append(color_test_values[0])
        else:
            if verbose==1:
                print(f"{one_group_ID} contains more then one COLOR to display wiht different names !")
            else:
                Stop_Function = True
                pass        
        
        
    # - check if everythign is in order
    if len(bx_colors)!=len(bx_names) and len(bx_names)!=len(bx_data):
        if verbose==True:
            print("Error: some data are missing or belong to different gorups, and can not be displayed as coherent bocplot")
        else:
            pass
    else:
        

        # ...............................................
        # data preparation - step.2 ordering
        # ...............................................    

        # find memdians and reorder 
        bx_medians = list()
        for i, d in enumerate(bx_data):
            bx_medians.append(np.median(d))   
            

        # ...
        ordered_data_df = pd.DataFrame({
            "bx_data":    bx_data,
            "bx_medians": bx_medians,
            "bx_names":   bx_names,
            "bx_colors":  bx_colors,
            "bx_id": bx_id,
            "bx_colors_dict_key":bx_colors_dict_key
        })
        ordered_data_df = ordered_data_df.sort_values("bx_medians", ascending=True)
        ordered_data_df = ordered_data_df.reset_index(drop=True)


        # ...............................................
        # boxplot
        # ...............................................    
        
        # ...............................................
        # boxplot,  - plt.boxplot(ordered_bx_data);
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        fig.suptitle(title, fontsize=title_fontsize)

        # add boxes,
        bx = ax.boxplot(ordered_data_df["bx_data"], 
                showfliers=True,                  # remove outliers, because we are interested in a general trend,
                vert=True,                        # boxes are vertical
                labels=ordered_data_df["bx_names"],           # x-ticks labels
                patch_artist=True,
                widths=0.3
        )
        ax.grid(ls="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticklabels(ordered_data_df["bx_names"], rotation=45, fontsize=xticks_fontsize, ha="right")
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=yticks_fontsize)
        ax.set_ylabel(yaxis_label, fontsize=axes_labels_fontsize)
        ax.set_xlabel(xaxis_label, fontsize=axes_labels_fontsize)
        ax.set_ylim(0,1.02)

 
        # add colors to each box individually,
        for i, j in zip(range(len(bx['boxes'])),range(0, len(bx['caps']), 2)) :
            median_color  ="black"
            box_color     = bx_color_dict[ordered_data_df.loc[:,"bx_colors_dict_key"].iloc[i]]

            # set properties of items with the same number as boxes,
            plt.setp(bx['boxes'][i], color=box_color, facecolor=median_color, linewidth=2, alpha=0.8)
            plt.setp(bx["medians"][i], color=median_color, linewidth=2)
            plt.setp(bx["fliers"][i], markeredgecolor="black", marker=".") # outliers

            # set properties of items with the 2x number of features as boxes,
            plt.setp(bx['caps'][j], color=median_color)
            plt.setp(bx['caps'][j+1], color=median_color)
            plt.setp(bx['whiskers'][j], color=median_color)
            plt.setp(bx['whiskers'][j+1], color=median_color)

         
        # ...............................................
        # set colors for xtick labels,  
        
        if paint_xticks==True:
            for i, xtick in enumerate(ax.get_xticklabels()):
                xtick.set_color(bx_color_dict[ordered_data_df["bx_colors_dict_key"].iloc[i]])
        else:
            pass
            
        # ...............................................            
        # legend,    
        if ordered_data_df["bx_names"].shape[0]>0:

            # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
            patch_list_for_legend =[]
            for i, m_name in enumerate(list(bx_color_dict.keys())):
                label_text = f"{m_name}"
                patch_list_for_legend.append(mpl.patches.Patch(color=bx_color_dict[m_name], label=label_text))

            # add patches to plot,
            fig.legend(
                handles=patch_list_for_legend, frameon=False, 
                scatterpoints=1, ncol=legend_ncols, 
                bbox_to_anchor=legend__bbox_to_anchor, fontsize=legend_fontsize)
            
            # ...............................................
            # create space for the legend
            fig.subplots_adjust(top=subplots_adjust_top)    
            # ...............................................
            
            
        # ...............................................
        # add line with baseline
        ax.axhline(baseline, lw=2, ls="--", color="dimgrey")
        ax.text(ordered_data_df.shape[0]+0.4, baseline+baseline_loc, baseline_title, ha="right", color="dimgrey", fontsize=yticks_fontsize)        

        
        # ...............................................
        # color patches behing boxplots, 
        patch_width = 1   # ie. 1 = grey patch for 1 and 1 break
        patch_color = "lightgrey"
        pathces_starting_x = list(range(0, ordered_data_df.shape[0], patch_width*2))
        # ...
        for i, sx in enumerate(pathces_starting_x):
            rect = plt.Rectangle((sx+0.5, 0), patch_width, 1000, color=patch_color, alpha=0.2, edgecolor=None)
            ax.add_patch(rect)        


        # color patches for styling the accuracy, 
        rect = plt.Rectangle((0,0), ordered_data_df.shape[0]*100, baseline, color="red", alpha=0.1, edgecolor=None)
        ax.add_patch(rect)          
        rect = plt.Rectangle((0,baseline), ordered_data_df.shape[0]*100, top_results-baseline, color="orange", alpha=0.1, edgecolor=None)
        ax.add_patch(rect)             
        rect = plt.Rectangle((0, top_results), ordered_data_df.shape[0]*100, 10, color="forestgreen", alpha=0.1, edgecolor=None)
        ax.add_patch(rect)            

    return fig
            
            









      
      
      
      

      
      
# Function, .......................................................................
def new_method_comparison_boxplot(*,                               
    title=f"Accuracy of models created with each method\n",
    data,  
    figsize=(45,15),
    # ...
    col_with_baseline,
    col_with_results,# df colname with values to display, eg: test_accuracy ...
    col_with_group_names,# df colname with values that will be displayed as names of each box (these do not have to be unique)
    col_with_group_ID,# df colname with values that will be grouped for separate boxes (must be unieque)
    col_with_group_colors,# df colname with values that will have different colors (colors can not be mixed within diffeent group_ID)
    col_with_pattern,# collumn that holds fule file name or its elements that indicated that a given results were made with full data or partial data
    full_data_pattern_list, 
                                # pattern in file name that will be used to udentify models create wiht full datasets, these will be pèlotted as scatterplots
    # ... colors       
    color_group_dict=None, 
    cmap="tab10",
    cmap_colors_from=0,
    cmap_colors_to=0.5,                               
    
    # .. legend
    legend_on=True,
    legend_title="class labels",
    legend__bbox_to_anchor=(0.7, 0.99),
    subplots_adjust_top = 0.8,
    legend_ncols=5,
    
    # .. baseline
    baseline_title = "",
    baseline_loc =-0.09,
    use_fixed_baselines = False,
    baseline_limit_list = [0.5, 0.9, 1.5], # the last one 
    baseline_color_list = ["red", "orange", "forestgreen"],
    
    # ... fontsizes
    title_fontsize=40,
    legend_fontsize=20,
    xticks_fontsize=8,
    yticks_fontsize=15,
    axes_labels_fontsize=20,
    
    # ... axies labels
    xaxis_label = "Method",
    yaxis_label = "Accuracy\n",
    paint_xticks=True,
    # ... scatterpoints for full models,
    add_full_models_markers =True, # if False, they are oging to be part of the barplots
    full_model_marker ="*",
    full_model_markersize=80,
    full_model_markercolor="black"
):


    """
        Nice function to create ngs-like boxplots for comparison of acc of differemnt model groups
        it is more generic version of the abofe function, 
    """

    
    # ...............................................
    # managment
    Stop_Function = False
    
    # ...............................................
    # data preparation - step.1 extraction
    # ...............................................

    # - extract unique values that will be searched, 
    unique_group_ID               = data.loc[:,col_with_group_ID].unique().tolist()
    unique_group_color_names      = data.loc[:,col_with_group_colors].unique().tolist()
    
    # - map colors onto color_groups_names
    if color_group_dict==None:
        bx_color_dict = create_class_colors_dict(
            list_of_unique_names=unique_group_color_names, 
            cmap_name=cmap, 
            cmap_colors_from=cmap_colors_from, 
            cmap_colors_to=cmap_colors_to
        )   
    else:
        bx_color_dict = color_group_dict
    
    # - lists with data for boxes, 
    'one item for one box in each'
    bx_data             = [] 
    bx_baseline         = []
    bx_names            = []
    bx_colors           = []
    bx_id               = []
    bx_colors_dict_key  = []
    bx_full_models      = []
    
    # - separate all boxes, and then find out what is the color and data associated with that box
    for one_group_ID in unique_group_ID:
        bx_id.append(one_group_ID)
        
        
        #### values
        
        # .. get the data and other columns for one box
        data_df_for_one_box = data.loc[data.loc[:, col_with_group_ID]==one_group_ID,:]
        
        # .. find baseline
        bx_baseline.append(data_df_for_one_box.loc[:, col_with_baseline].max())
        
        # .. find out, data, name and color to display
        # .... num. data ....
        bx_data.append(data_df_for_one_box.loc[:,col_with_results].values) # np.array
        
        # .. scatterpoints with the top performing models
        for p_i, pat in enumerate(full_data_pattern_list):
            arr_temp = data_df_for_one_box.summary_table__file_name.str.contains(pat).values
            if p_i==0:
                full_model_pos = arr_temp
            else:
                full_model_pos[arr_temp]=True 
        if full_model_pos.sum()==0:
            bx_full_models.append(data_df_for_one_box.loc[:,col_with_results].max()) 
            # here I will use only one value, full models were alwaasws having higher acc then the rest
        else:     
            bx_full_models.append(data_df_for_one_box.loc[full_model_pos.tolist(),col_with_results].values)     

            
        #### colors and labels, 
        
        # .... labels .....
        one_bx_label = data_df_for_one_box.loc[:,col_with_group_names].unique().tolist()
        if len(one_bx_label)==1:
            bx_names.append(one_bx_label[0]) # np.array
        else:
            if verbose==1:
                print(f"{one_group_ID} contains more then one group to display wiht different names !")
            else:
                Stop_Function = True
                pass
            
            
        # .... colors ....
        one_box_color = data_df_for_one_box.loc[:,col_with_group_colors].map(bx_color_dict).iloc[0]
        color_test_values = data_df_for_one_box.loc[:,col_with_group_colors].unique().tolist()
        if len(color_test_values)==1:
            bx_colors.append(one_box_color) # np.array
            bx_colors_dict_key.append(color_test_values[0])
        else:
            if verbose==1:
                print(f"{one_group_ID} contains more then one COLOR to display wiht different names !")
            else:
                Stop_Function = True
                pass        
        
        
    # - check if everythign is in order
    if len(bx_colors)!=len(bx_names) and len(bx_names)!=len(bx_data):
        if verbose==True:
            print("Error: some data are missing or belong to different gorups, and can not be displayed as coherent bocplot")
        else:
            pass
    else:
        

        # ...............................................
        # data preparation - step.2 ordering
        # ...............................................    

        # find memdians and reorder 
        bx_medians = list()
        for i, d in enumerate(bx_data):
            bx_medians.append(np.median(d))   
            

        # ...
        ordered_data_df = pd.DataFrame({
            "bx_data":    bx_data,
            "bx_medians": bx_medians,
            "bx_names":   bx_names,
            "bx_colors":  bx_colors,
            "bx_id":      bx_id,
            "bx_colors_dict_key":bx_colors_dict_key,
            "bx_baseline": bx_baseline,
            "bx_full_models":bx_full_models
        })
        ordered_data_df = ordered_data_df.sort_values("bx_medians", ascending=True)
        ordered_data_df = ordered_data_df.reset_index(drop=True)


        # ...............................................
        # boxplot
        # ...............................................    
        
        # ...............................................
        # boxplot,  - plt.boxplot(ordered_bx_data);
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        fig.suptitle(title, fontsize=title_fontsize)

        # add boxes,
        bx = ax.boxplot(ordered_data_df["bx_data"], 
                showfliers=True,                  # remove outliers, because we are interested in a general trend,
                vert=True,                        # boxes are vertical
                labels=ordered_data_df["bx_names"],           # x-ticks labels
                patch_artist=True,
                widths=0.3
        )
        
        if add_full_models_markers==True:
            for xpos in np.arange(ordered_data_df.shape[0]):
                ypos = ordered_data_df.bx_full_models.iloc[xpos]

                # this may be one or many values, 
                if isinstance(ypos, float):
                    ax.scatter(x=xpos, y=ypos ,color=full_model_markercolor, s=full_model_markersize,  marker=full_model_marker, zorder=100)
                else:
                    xposs = np.array([xpos]*ypos.shape[0])
                    ax.scatter(x=xposs, y=ypos ,color=full_model_markercolor, s=full_model_markersize,  marker=full_model_marker, zorder=100)
        else:
            pass
        
        ax.grid(ls=":", lw=0.001)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticklabels(ordered_data_df["bx_names"], rotation=45, fontsize=xticks_fontsize, ha="right")
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=yticks_fontsize)
        ax.set_ylabel(yaxis_label, fontsize=axes_labels_fontsize)
        ax.set_xlabel(xaxis_label, fontsize=axes_labels_fontsize)
        ax.set_ylim(0,1.02)

 
        # add colors to each box individually,
        for i, j in zip(range(len(bx['boxes'])),range(0, len(bx['caps']), 2)) :
            median_color  ="black"
            box_color     = bx_color_dict[ordered_data_df.loc[:,"bx_colors_dict_key"].iloc[i]]

            # set properties of items with the same number as boxes,
            plt.setp(bx['boxes'][i], color=box_color, facecolor=median_color, linewidth=2, alpha=0.8)
            plt.setp(bx["medians"][i], color=median_color, linewidth=2)
            plt.setp(bx["fliers"][i], markeredgecolor="black", marker=".") # outliers

            # set properties of items with the 2x number of features as boxes,
            plt.setp(bx['caps'][j], color=median_color)
            plt.setp(bx['caps'][j+1], color=median_color)
            plt.setp(bx['whiskers'][j], color=median_color)
            plt.setp(bx['whiskers'][j+1], color=median_color)

         
        # ...............................................
        # set colors for xtick labels,  
        
        if paint_xticks==True:
            for i, xtick in enumerate(ax.get_xticklabels()):
                xtick.set_color(bx_color_dict[ordered_data_df["bx_colors_dict_key"].iloc[i]])
        else:
            pass
            
        # ...............................................            
        # legend,    
        if ordered_data_df["bx_names"].shape[0]>0:

            # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
            patch_list_for_legend =[]
            for i, m_name in enumerate(list(bx_color_dict.keys())):
                label_text = f"{m_name}"
                patch_list_for_legend.append(mpl.patches.Patch(color=bx_color_dict[m_name], label=label_text))

            if legend_on==True:    
                # add patches to plot,
                leg = fig.legend(
                    handles=patch_list_for_legend, frameon=False, title=legend_title,
                    scatterpoints=1, ncol=legend_ncols, 
                    bbox_to_anchor=legend__bbox_to_anchor, fontsize=legend_fontsize)
                plt.setp(leg.get_title(),fontsize=legend_fontsize)#'xx-small')
            else:
                pass
                
            # ...............................................
            # create space for the legend
            fig.subplots_adjust(top=subplots_adjust_top)    
            # ...............................................
            
            
        # ...............................................
        # color patches behing boxplots, 
        patch_width = 1   # ie. 1 = grey patch for 1 and 1 break
        patch_color = "lightgrey"
        pathces_starting_x = list(range(0, ordered_data_df.shape[0], patch_width*2))
        # ...
        for i, sx in enumerate(pathces_starting_x):
            rect = plt.Rectangle((sx+0.5, 0), patch_width, 1000, color=patch_color, alpha=0.3, edgecolor=None)
            ax.add_patch(rect)        


        # ...............................................
        # BASELINE color patches, 
        if use_fixed_baselines==False:
            for i, one_baseline in enumerate(ordered_data_df.bx_baseline.values.tolist()):
                # color patches for styling the accuracy, 
                rect = plt.Rectangle((i+0.5,0), width=1, height=one_baseline, color=baseline_color_list[0], alpha=0.1, edgecolor=None)
                ax.add_patch(rect)              
                rect = plt.Rectangle((i+0.5,one_baseline), width=1, height=(1-one_baseline)/2, color=baseline_color_list[1], alpha=0.1, edgecolor=None)
                ax.add_patch(rect)  
                rect = plt.Rectangle((i+0.5,one_baseline+(1-one_baseline)/2), width=1, height=1000, color=baseline_color_list[2], alpha=0.1, edgecolor=None)
                ax.add_patch(rect)  

        else:       
            rect = plt.Rectangle((0,0), ordered_data_df.shape[0]*100, baseline_limit_list[0], color=baseline_color_list[0], alpha=0.1, edgecolor=None)
            ax.add_patch(rect)          
            rect = plt.Rectangle((0,baseline_limit_list[0]), ordered_data_df.shape[0]*100, baseline_limit_list[1]-baseline_limit_list[0], color=baseline_color_list[1], alpha=0.1, edgecolor=None)
            ax.add_patch(rect)             
            rect = plt.Rectangle((0,baseline_limit_list[1]), ordered_data_df.shape[0]*100, 1000, color=baseline_color_list[2], alpha=0.1, edgecolor=None)
            ax.add_patch(rect)               
            # ...............................................
            # add line with baseline
            ax.axhline(baseline_limit_list[0], lw=2, ls="--", color="dimgrey")
            ax.text(ordered_data_df.shape[0]+0.4, baseline_limit_list[0]+baseline_loc, baseline_title, ha="right", color="dimgrey", fontsize=yticks_fontsize)        
       
            










