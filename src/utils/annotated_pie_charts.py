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
from src.utils.tools_for_plots import create_class_colors_dict



# Function ...........................................................................
# new
def annotated_pie_chart_with_class_and_group(*, 
                                             # data
                                             classnames, 
                                             groupnames=None, 
                                             
                                             # general fig/plot aestetics
                                             title=None, 
                                             title_ha="right",
                                             title_fontsize_scale=1,
                                             class_colors=None, 
                                             groupname_colors=None,
                                             class_colors_cmap="tab20",
                                             cmap_colors_from =0, 
                                             cmap_colors_to =1,
                                             
                                             # fig size & layout
                                             figsze_scale=1, 
                                             figwidth_scale=1, 
                                             figheight_scale=1,                                      
                                             n_subplots_in_row=3, 
                                             subplots_adjust_top=0.9, 
                                             tight_lyout=False, 
                                             
                                             # legend, fonts and additional text
                                             legend=True, 
                                             legend_loc="center", 
                                             legend_ncol=6, 
                                             legend_fontsize_scale=1, 
                                                
                                             # piecharts on each subplot
                                             ax_title_fontcolor=None, 
                                             ax_title_fonsize_scale=1, 
                                             wedges_fontsize_scale=1, 
                                             add_group_name_to_each_pie=True, 
                                             add_group_item_perc_to_numbers_in_each_pie=True, 
                                             mid_pie_circle_color="lightgrey", 
                                             
                                             verbose=False
                                            ):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          function crerates annotated pie charts with empty center, 
                            annotations, have name of the class, number of instances and pecentage of instances, 
                            in the total population
                            optionally, the functions can take second argument, groupnames, of the same lenght as cvlassnames, 
                            if used, groupnames, will be used to create separate annotated pie chart, for each uniqwue groupname, 
                            with groupname in the middle of the pie chart.

        # Inputs
        .......................     ...........................................................................
        . classnames                : list, with repeated instances of items that will be counted and presented as classes on pie chart
        . groupnames                : list, with repeated instances of groupnames, used to create separate pie charts, 
                                      default=None, 
        . title                     : str, title above the figure, with all images, 
        . verbose                   : bool, default=False
        . class_colors              : dictionary,  {str <"class_name">: str <"color">} 
                                      used, to color pie classes on pie chart
        . groupname_colors          : dictionary,  {str <"group_name">: str <"color">}
                                      used to color group name, in the middle of pie chart - a gorupname, 
                                     CAUTION: colors and class names must be unique !
        # Returns
        .......................     ...........................................................................
        Matplotlib figure, 
        
        # Notes
        Pie chart idea taken from
        https://matplotlib.org/3.1.0/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py
        
        
        
    """

    # small correction, on error i did with names while creasting this function
    img_classnames = classnames
    img_groupnames = groupnames
    
    
    # .................................................................
    # DATA PREPARATION,  
    if img_groupnames==None: 
        img_groupnames =  ["one group only"]*len(img_classnames)
        if verbose==True: 
            print("img_groupname were not specified ...  all images will be plotted one after anothe, as they woudl belong to one group, cluster, ...")
        else: 
            pass
    else: 
        pass
    # ...
    groups_to_plot = pd.Series(img_groupnames).unique().tolist()


    # .................................................................
    # FIGURE PARAMETERS, 
    
    # figsize aand subplot number 
    if len(groups_to_plot)<=n_subplots_in_row:
        fig_nrows = 1
        fig_height = 4.5
        # ...
        fig_ncols = len(groups_to_plot)
        figsize_width = fig_ncols*5*figsze_scale
    
    if len(groups_to_plot)>n_subplots_in_row:
        fig_nrows = int(np.ceil(len(groups_to_plot)/n_subplots_in_row))
        fig_height = fig_nrows*4
        # ...
        fig_ncols = n_subplots_in_row
        figsize_width = 5*n_subplots_in_row*figsze_scale
    # ..
    fig_size = (figsize_width*figwidth_scale, fig_height*figheight_scale)
    
    # ..
    title_fonsize    = 40
    ax_title_fonsize = title_fonsize*0.4*ax_title_fonsize_scale
    wedges_fontsize  = title_fonsize*0.25*wedges_fontsize_scale
    
    # pie dimensions, 
    pie_size_scale   = 0.8 # proportion of the plot in x,y dimensions
    pie_width_proportion = 0.33

    # class colors, - chnages because I added legend that looks really nice, 
    if class_colors==None:
        class_colors = create_class_colors_dict(
            list_of_unique_names = pd.Series(img_classnames).unique().tolist(),
            cmap_name=class_colors_cmap, 
            cmap_colors_from = cmap_colors_from, 
            cmap_colors_to = cmap_colors_to
            )
    else:
        pass
    
    # .................................................................
    # FIGURE,     
    
    # Figure and axes, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(ncols=fig_ncols, nrows=fig_nrows, figsize=(fig_size), facecolor="white")

    # .. add title, 
    if title!=None: 
        fig.suptitle(title, fontsize=title_fonsize*0.6*title_fontsize_scale, color="black", ha=title_ha)
    else: 
        pass

    if len( groups_to_plot)==1:
        axss = [axs]
    else:
        axss = axs.flat
    
    
    # .. create each subplot with pie annotated chart, 
    for ax_i, ax in enumerate(axss):
    
    
        if ax_i>=len(groups_to_plot):
            # empty, plot, so clear axis,  and keep it that way, 
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)            
        
        else:

            # set group name for a given subplot, 
            one_groupname = groups_to_plot[ax_i]


            # clear axis, - saves a bit of space,  
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # select classnames 
            s = pd.Series(img_classnames).loc[pd.Series(img_groupnames)==one_groupname]
            s_item_number = s.shape[0]
            s = s.value_counts()

            # find colors for pie chart
            if class_colors!=None:
                one_group_pie_colors = list()
                for j, cn in enumerate(s.index.values.tolist()):
                    one_group_pie_colors.append(class_colors[cn])
            else:
                one_group_pie_colors=None

            # create description for each calls with its percentage in df column
            pie_descr = list(s.index)
            data      = [float(x) for x in list(s.values)]
            pie_descr = [f"{y}: {str(int(x))} ({str(np.round(x/np.sum(data)*100))}%)" for x,y in zip(data, pie_descr)]
            # pie
            wedges, texts = ax.pie(
                data, 
                wedgeprops=dict(width=pie_width_proportion*pie_size_scale),  # Caution, here must be two numbers !!!
                radius=pie_size_scale,
                startangle=-60, 
                counterclock=False,
                colors=one_group_pie_colors
            )

            # params for widgets
            bbox_props = dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="k", lw=1, alpha=0.3)
            kw = dict(arrowprops=dict(arrowstyle="->"),
                      bbox=bbox_props, zorder=10, va="center", fontsize=wedges_fontsize)

            # add widgest to pie chart with pie descr
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))*pie_size_scale
                x = np.cos(np.deg2rad(ang))*pie_size_scale
                # ...
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                # ...
                ax.annotate(pie_descr[i], xy=(x, y), xytext=(1*np.sign(x), 1.4*y),
                            horizontalalignment=horizontalalignment, **kw)

            # add groupname, in the center of pie chart, 
            
            # .. if, available set color for groupname
            if groupname_colors==None:
                if ax_title_fontcolor==None:
                    font_color="black"
                else:
                    font_color=ax_title_fontcolor
                # ensure that anythign is visible, eg donth have black text on black bacground    
                if mid_pie_circle_color==font_color:
                    font_color="white"
                else:
                    pass
                patch = plt.Circle((0, 0), (pie_size_scale-pie_width_proportion), zorder=0, alpha=1, color=mid_pie_circle_color)
                ax.add_patch(patch)
            else:
                if ax_title_fontcolor==None:
                    font_color="white"
                else:
                    font_color=ax_title_fontcolor
                # ensure that anythign is visible, eg donth have black text on black bacground
                if groupname_colors[one_groupname]==font_color:
                    font_color="white"
                else:
                    pass                   
                one_groupname_color=groupname_colors[one_groupname]
                patch = plt.Circle((0, 0), (pie_size_scale-pie_width_proportion), zorder=0, alpha=1, color=one_groupname_color)
                ax.add_patch(patch)
            
            # .. add group name with larger font, and number associated with that group (item count and % in total dataset)
            if len(groups_to_plot)>1 or add_group_name_to_each_pie==True:
                font = FontProperties()
                # ..
                font.set_weight("bold")
                font.set_size(ax_title_fonsize)
                ax.text(0, 0, one_groupname, fontsize=ax_title_fonsize, ha="center", color=font_color, fontproperties=font)
                # ...
                font.set_size(wedges_fontsize)
                if add_group_item_perc_to_numbers_in_each_pie==True:
                    ax.text(0, -0.2, f"{s_item_number}, ({np.round((s_item_number/len(img_classnames)*100),1)}%)", 
                          fontsize=wedges_fontsize, ha="center", fontproperties=font, color=font_color)           
                else:
                    ax.text(0, -0.2, f"{s_item_number}", 
                          fontsize=wedges_fontsize, ha="center", fontproperties=font, color=font_color)           



    # .............................................................................
    # LEGEND 
    if legend==True:
        # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
        patch_list_for_legend =[]
        count_items = 0
        for i, cl_name in enumerate(list(class_colors.keys())):
            cl_color = class_colors[cl_name]
            patch_list_for_legend.append(mpatches.Patch(color=cl_color, label=cl_name))

        # add patches to plot,
        fig.legend(
            handles=patch_list_for_legend, 
            frameon=False, 
            scatterpoints=1, ncol=legend_ncol, 
            fontsize=ax_title_fonsize*0.8*legend_fontsize_scale,
            loc=legend_loc
        )                
    else:
        pass

    # .............................................................................
    # END   
    
    if tight_lyout==True:
        plt.tight_layout()
    else:
        pass
    plt.subplots_adjust(top=subplots_adjust_top)
    plt.show();
    
    
    
    
    
    
# Function ...........................................................................

def prepare_img_classname_and_groupname(*, data_for_plot, groupname_prefix="Cluster ", number_of_img_examples=100, plot_img_from=None, plot_img_to=None):
    """
        Helper function to get img class name and group name for annotated pie charts, 
        from results obtained after images examples were plotted with plot_img_examples_from_dendrogram()
    """

    # set idx 
    if plot_img_from!=None and plot_img_to!=None:
        img_idx = data_for_plot['img_order_on_dedrogram'][plot_img_from:plot_img_to].tolist()
    else:
        temp = np.unique(np.floor(np.linspace(0, data_for_plot['batch_labels'].shape[0], number_of_img_examples, endpoint=False)).astype(int))
        img_idx = data_for_plot['img_order_on_dedrogram'][temp]

    # find idx if images in batch_labels, but ordered as on dendrogram, 
    selected_df_for_plot = data_for_plot['batch_labels'].loc[img_idx,:]
    selected_df_for_plot.reset_index(drop=False, inplace=True)

    # preapre df with selected data for the plot¨
    img_classname = selected_df_for_plot.classname.values.tolist()
    img_groupname = ["".join([groupname_prefix,str(x)]) for x in selected_df_for_plot.loc[:, "dendrogram_clusters"].values.tolist()]
    
    return img_classname, img_groupname



