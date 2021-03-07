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


############################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





# Function, ......................................................................................................

def pca_analyis_with_plots(*, path, dataset_name, subset_names, module_names, scale_data=False, class_colors=None,
                              title=None, fig_nrows=1, fig_ncols=1, figsize=(3,3), title_fontsize=25,
                              markersize=10, markeralpha=0.5, subplots_adjust_top=0.7, verbose=False):


    # dict to collect trained pca, 
    pca_models_dict = dict()

    
    # Create a figure, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(nrows=fig_nrows, ncols=fig_ncols, facecolor="white", figsize=figsize)
    if title!=None:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        pass
      
    # ... in case you have only one subplot, 
    if fig_ncols*fig_nrows==1:
        axss = [axs]
    else:
        axss = axs.flat  

    # Load, transform, and plot dataset create with each module, 
    for ei, ax in enumerate(axss):

        if ei>=len(module_names):
            # empty, plot, with clear axis, eg. at the end of the plot, 
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)       
        
        else:
          # Load the data, .....................................
          module_name = module_names[ei]
          os.chdir(path)

          # .. fing any logfile created while saving img files, 
          logfiles = []
          for file in glob.glob(f"{''.join([module_name,'_',dataset_name])}*_logfile.csv"):
                      logfiles.append(file)

          # .. load batch labels,  
          encoded_img_batch, batch_labels = load_encoded_imgbatch_using_logfile( 
              logfile_name=logfiles[0], 
              load_datasetnames=subset_names)

          # Standarization , ...................................
          if scale_data==True:
              # ... Create Standard scaler
              scaler = StandardScaler()

              # ...Rescale data - Use scaler.transform(X) to new data!
              encoded_img_batch = scaler.fit_transform(encoded_img_batch)

              if verbose==True:
                  print("- Caution -  You are standarizing the featzures")
              else:
                  pass
          else:
              pass    

          # PCA, ..............................................

          # Train PCA model
          pca = PCA(n_components=None) # it will use max nr of components == nr of features in dataset !
          pca.fit(encoded_img_batch, y=None) # Unsupervised learning, no y variable
          X_pca = pca.transform(encoded_img_batch) # Project data onto the first two components

          # .. store the result for the other functions, 
          pca_models_dict[module_name] = pca


          # Plot each class separately, ......................
          if class_colors!=None:
              for one_class_name in list(class_colors.keys()):
                  # find items, and select color for points
                  idx = (batch_labels.classname.values == one_class_name)
                  one_class_color = class_colors[one_class_name]

                  # Plot their components
                  ax.scatter( 
                      X_pca[idx, 0],                 # if more axes woudl be used, 
                      X_pca[idx, 1],                 #. then X_2d woudl have more collumns, 
                      label=one_class_name,
                      s=markersize,
                      color=one_class_color,
                      alpha=markeralpha
                  )
          else:
              ax.scatter( X_pca[:,0], X_pca[:, 1], label="data", s=markersize, color="steelblue", alpha=markeralpha )            

          # ... Labels and legend
          ax.set_title(module_name)
          ax.set_xlabel('1st component - principial axes')
          ax.set_ylabel('2nd component')
          handles, labels = ax.get_legend_handles_labels()
          # ...
          ax.xaxis.set_major_locator(plt.MaxNLocator(6))
          ax.yaxis.set_major_locator(plt.MaxNLocator(6))
          # ...
          ax.spines["right"].set_visible(False) # and below, remove white border, 
          ax.spines["top"].set_visible(False)

          # ensure that ax box is square
          #ax.set_aspect('equal', 'box')

          # Legend, ........................................
          l = ax.legend(handles, labels, loc=(1, 0), fontsize=10, frameon=False) # bbox_to_anchor = (0,-0.1,1,1)
          l.set_title('Class:',prop={'size':10}) # otherwise, I can not control fontsize of the title,

    # layout, 
    fig.tight_layout()
    fig.subplots_adjust(top=subplots_adjust_top)
    plt.show();
    
    return pca_models_dict






# Function, ......................................................................................................

def pca_scree_plots(*, pca_models_dict, fig_nrows=1, fig_ncols=1, 
                    title=None,figsize=(4, 3), ylim_max=0.11, x_overhangs=20,  
                    title_fontsize=25, subplots_adjust_top=0.7, annotate_lines=True):
    """
        Function that creates scree plot for pca reults stored in dictionary, 
        each subplot, has two axes, one with proportion of varinace explained by each principial components, shown as barplot,
        and second y axis, with different scale (0-100%) showing cumulatice pve as % of explained vaiance, using stepplot, 
        Caution: you must manually sset number of columns and rows on the figure for setting subplot nr right, 
    
            # Input,
            . pca_models_dict: dictionary, with trained and fitted pca models, 
                               dict.key will be used as subplot title, 
            . ....
            . title         : str, 
            . ylim_max      : between 0, and 1.1
            . x_overhangs   : int, how much the x axis shodul be longer on both sites arround the plot, 
                              to allow visualising all the lines, 
                              from each site it will be xmin-x_overhangs/2, and xmax+x_overhangs/2
            . ....
            . fig_nrows     : int, fig_nrows*fig_ncols must be => pca_models_dict size, 
            . fig_ncols     : int, see fig_nrows, 
            . figsize       : tuple, with two integerss, 
            . ....
            . annotate_lines: bool, is True, guidlines, showing the number of princial axes that allow explanining 
                              10%, 20% 50% 90% and 99% or 100%  will be annotated with text, 
                
            # returns, 
            . matplotlib figure, 
                
            
    """

    # Create a scree plot, 
    module_names = list(pca_models_dict.keys())

    # Create a figure, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(nrows=fig_nrows, ncols=fig_ncols, facecolor="white", figsize=figsize)
    if title!=None:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        pass

    # ... in case you have only one subplot, 
    if fig_ncols*fig_nrows==1:
        axss = [axs]
    else:
        axss = axs.flat  

    # Load, transform, and plot dataset create with each module, 
    for i, ax in enumerate(axss):

        if i>=len(module_names):
            # empty, plot, with clear axis, eg. at the end of the plot, 
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)       
        
        else:      
            module_name = module_names[i]
            

            # Part 1. Data preparation, 

            # .. extract the % of the variance represented by each principial compenent 
            pve = pca_models_dict[module_name].explained_variance_ratio_

            # .. calculate cumulative sum
            pve_cumsum = np.cumsum(pve)

            # how many components explain 10%, 20%, 90% and 100% of the varinace in the features ?
            #pve_th = np.linspace(0.1, 1, 10, endpoint=True) # - these will be displayed as lines, 
            pve_th = np.array([0.1, 0.2, 0.5, 0.9, 1])
            pve_th_feature_nr = list()
            for j, p in enumerate(pve_th):
                try:
                    pve_th_feature_nr.append(np.where(pve_cumsum>=p)[0][0])
                except:
                    # if not availbel, take the last one, 
                    pve_cumsum_idx = np.where(pve_cumsum<p)[0][-1]
                    pve_th_feature_nr.append(pve_cumsum_idx)
                    # .. and update, corresponding pve_th
                    pve_th[j]= pve_cumsum[pve_cumsum_idx]


            # Part 2. scree-plot, 

            # a) barplot on ax, 
            bar_color = "blue"

            # barplot, with varinace explained by each p. component
            xcordinates = np.arange(1, len(pve) + 1) # 1,2,..,n_components
            ax.bar(xcordinates, pve, color=bar_color, edgecolor=bar_color)  
            # .....
            ax.set_ylim(0, ylim_max) 
            ax.set_xlim(-x_overhangs/2, len(pve)+x_overhangs/2)
            # ......
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["top"].set_visible(False)
            ax.set(
                title=module_name,
                xlabel="Principial component number")

            ax.set_ylabel('proportion of\nvariance explained(pve)')


            # b) step plot on ax2, 

            # instantiate a second axes that shares the same x-axis
            ax2 = ax.twinx() 
            ax2.set_ylim(0,1.1)

            # step line with cumulative sum
            ax2.step(
                xcordinates+0.5, # 1.5,2.5,..,n_components+0.5
                pve_cumsum, # Cumulative sum
                color="red",
                label='cumulative pve',
                lw=0.5
            )

            # title ad aestetists, 
            ax2.set(
                title=module_name,
                xlabel="Principial components")
            ax2.set_ylabel('cumulative percentage\nof explained variance', color="red")
            ax2.tick_params(axis='y')
            ax2.spines["top"].set_visible(False)

            # legend
            #ax2.legend(frameon=True, loc="center right")

            # set yticks, for ax2
            ax2.set_yticks(np.linspace(0.1, 1, 10, endpoint=True))
            ax2.set_yticklabels(   
                [ f"{str(int(x*100))}%" for x in np.linspace(0.1, 1, 10, endpoint=True).tolist()],
                color="red"
            )

            # add guidlineslines, to alow estimating how many principial components allow explaning 10%, 20% etc.. of varinace in data
            linecolor = "grey"

            # add vertical lines, 
            for j, xline in enumerate(pve_th_feature_nr):
                ymax = pve_cumsum[xline]*10/11 # multiplied by 10/11 because ymax in axhline is on axis cordinates from 0 to 1.
                ax2.axvline(xline, ymin=0, ymax=ymax, lw=0.5, color=linecolor, ls=":")

            # add horizontal lines, 
            for j,(p, xline) in enumerate(zip(pve_th, pve_th_feature_nr)):
                plot_x_size = (len(pve)+20)
                xmin = xline/plot_x_size
                ax2.axhline(p, xmin=xmin, xmax=plot_x_size, lw=0.5, color=linecolor, ls=":")


            # add annotastion, with number of pca axes that allow explainign certain % of varinace, 
            if annotate_lines==True:
                for j,(p, xline) in enumerate(zip(pve_th, pve_th_feature_nr)):
                    if (xline+1)==1:
                        annotation_end="p.component"
                    else:
                        annotation_end="p.components"
                    line_annotation_text = f"{int(p*100)}% explained with {xline+1} {annotation_end}"
                    # ...
                    text_x_position = len(pve)*0.18
                    text_y_position = (pve_cumsum[xline])
                    # ...
                    ax2.text(text_x_position, text_y_position, line_annotation_text, fontsize=8, color=linecolor, ha="left")
            else:
                pass
                     
    plt.tight_layout()    
    plt.subplots_adjust(top=subplots_adjust_top)
    plt.show();
    
    
    
    

    
    
# Function, ......................................................................................................   

def pca_pve_vs_variance_scatterplots(*, pca_models_dict, path, dataset_name, subset_names, module_names, scale_data=False, pca_axes_max_nr=None,
                                     title=None, fig_nrows=1, fig_ncols=1, figsize=(3,3), upper_ylim=0.2, scale_for_markers=20,
                                     add_noise=False, xlimits=None, title_fontsize=25, subplots_adjust_top=0.7, 
                                     verbose=False):    

    
    
    
    
    # plot configuration,  
    markercolor="steelblue"
    markeralpha=0.2
    
    # dict to store results,
    rdct = dict()
    
    
    # Figure, ........................................................
    
    # Create a figure, and then load datasets, for each subplot, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(nrows=fig_nrows, ncols=fig_ncols, facecolor="white", figsize=figsize)
    if title!=None:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        pass
    
    # ... in case you have only one subplot, 
    if fig_ncols*fig_nrows==1:
        axss = [axs]
    else:
        axss = axs.flat  

    # Load, transform, and plot dataset create with each module, 
    for i, ax in enumerate(axss):

        if i>=len(module_names):
            # create an empty subplot, with clear axis, eg. at the end of the plot, 
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)       
        
        else:      
            module_name = module_names[i]

            # Load the data, .....................................
        
            # .. fing any logfile created while saving img files, 
            logfiles = []
            for file in glob.glob(f"{''.join([module_name,'_',dataset_name])}*_logfile.csv"):
                logfiles.append(file)

            # .. load batch labels,  
            encoded_img_batch, batch_labels = load_encoded_imgbatch_using_logfile( 
                logfile_name=logfiles[0], 
                load_datasetnames=subset_names)    

            
            # Standarization , ...................................
            if scale_data==True:
                # ... Create Standard scaler
                scaler = StandardScaler()

                # ...Rescale data - Use scaler.transform(X) to new data!
                encoded_img_batch = scaler.fit_transform(encoded_img_batch)
                
                if verbose==True:
                    print("- Caution -  You are standarizing the featzures")
                    print("please make sure, to use pca_models_dict that was alse created with scaled input data !!!")
                else:
                    pass
            else:
                pass
 
            # data preparation for plot, ..........................
            
            # select how many princilia components to check
            if pca_axes_max_nr==None:
                idx_max = encoded_img_batch.shape[0]
            else:
                idx_max = pca_axes_max_nr

            # .. x-axis, (array where rows: principial components, cols: features,)
            "cols are: 1:pve, 2:variance in the feature, 3:feature corr with paxis"
            pca_components_corr_with_features = pca_models_dict[module_name].components_[0:idx_max,:]
            pve = pca_models_dict[module_name].explained_variance_ratio_[0:idx_max]
            feature_var = encoded_img_batch.var(axis=0)
            # ... from time to time, standard scaller generate very small negative values, thus:
            feature_var = feature_var.clip(0) # these will be turned to zero, 
            
            # prepare array_for_plot
            for row_counter, pve_i in enumerate(pve.flatten().tolist()):
                pve_x_features = np.array([pve_i]*feature_var.shape[0])
                temp_arr = np.c_[pve_x_features, feature_var, pca_components_corr_with_features[row_counter,:].flatten()]
                # ...
                if add_noise==True:
                    # adding small gausian noise to varinace, to better visualize the points, 
                    #.       - done in order to allow seeing points in standarized datsets
                    random_noise_to_x = np.random.rand(temp_arr.shape[0])*np.random.choice(np.array([-1,1]), size=temp_arr.shape[0], replace=True)/100
                    random_noise_to_y = np.random.rand(temp_arr.shape[0])*np.random.choice(np.array([-1,1]), size=temp_arr.shape[0], replace=True)/100
                    temp_arr[:,1] = temp_arr[:,1]+random_noise_to_x
                    temp_arr[:,0] = temp_arr[:,0]+random_noise_to_y
                else:
                    pass
                # ...
                if row_counter==0:
                    array_for_plot = temp_arr
                else:
                    array_for_plot = np.r_[array_for_plot, temp_arr]

            # plot, ..............................................
            ax.scatter(
                array_for_plot[:,1],    # x - feature variance, 
                array_for_plot[:,0],    # y - pve on axis, 
                s=((np.absolute(array_for_plot[:,2]))*scale_for_markers)**2, # correlation between principial component and a given feature,
                color=markercolor, 
                alpha=markeralpha,
                marker="o"
            )
            
            # ... title
            ax.set_title(module_name)
            
            # ... axes, 
            ax.set_ylim(0, upper_ylim)
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["top"].set_visible(False)            
            if xlimits==None:
                pass
            else:
                ax.set(xlim=xlimits)
            
            
            # ... labels,
            ax.set_xlabel("Variance in each feture")
            ax.set_ylabel('proportion of\nvariance explained(pve)')
 

            # legend, ..........................................
            legend_point_size = [1, 0.5, 0.2, 0.1]
            for lps in legend_point_size:
                ax.scatter([], [], c="red", alpha=markeralpha, s=(np.absolute(lps)*scale_for_markers)**2, label=f"abs(r)={lps}")
            l = ax.legend(scatterpoints=1, frameon=False, labelspacing=1.5, fontsize=8, loc=(1,0.1))
            l.set_title("Correlation between\nprincipial component\nand a given feature",prop={'size':8})
        
            # collect the results, for other plots, 
            rdct[module_name] = array_for_plot


    plt.tight_layout()    
    plt.subplots_adjust(top=subplots_adjust_top)
    plt.show();
    
    return rdct
  
  
  
  
  
  
  
  
  
# V2 ####################################################################### 2020.12.11






# Function, .............................................................
# new 2020.12.12
def pca_scatterplot(*, 
    title=None,
    pca_data_dict,
    class_colors=None,
    figsize_per_subplot=(5,4),
    title_fontsize=25,
    markersize=10, 
    markeralpha=0.5, 
    adjust_markeralpha=True,
    min_adjusted_markeralpha=0.2,
    subplots_adjust_top=0.7
    ):
    '''
        helper fnction for scatter plot using results generated 
        with mutiple labelleing and tf module for feature extraction 
        from the same dataset. all rquired data for plot 
        are stored in pca_data_dict
        all plots are in one line (nrow=1, ncol=plot nr)
        
        . figsize_per_subplot=(5,4) - first digit is mutiplied by the number of subplots
        ....                        - second int, is a figure heigh
        
        
        # Notes:
        . Unpacs following items from the dict
         # data for one subplot, 
            X_pca          = pca_data_dict[dict_key]["pca__example_extracted_features"]
                             # PCA axes for the datase (we use 1, and 2 for the plot)
            y              = pca_data_dict[dict_key]["example_label_for_each_img"]
                             # labels for each image
            y_unique_list  = y.unique().tolist() 
            class_colors   = pca_data_dict[dict_key]["example_class_labels_colors_dict"] 
                             # y_unique is a key, color is a value
        
    '''

    # Params, 
    fig_ncols = len(pca_data_dict)
    fig_nrows = 1
    figsize = (figsize_per_subplot[0]*fig_ncols, figsize_per_subplot[1])

    
    # Make figure, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(nrows=fig_nrows, ncols=fig_ncols, facecolor="white", figsize=figsize)
    if title!=None:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        pass
      
    # axes must but a list - in case you have only one subplot, 
    if fig_ncols*fig_nrows==1:
        axss = [axs]
    else:
        axss = axs.flat  

        
    # Load, transform, and plot dataset create with each module, 
    for ei, (ax, dict_key) in enumerate(zip(axss, list(pca_data_dict.keys()))):
    
        # data for one subplot, 
        X_pca          = pca_data_dict[dict_key]["pca__example_extracted_features"]
        y              = pca_data_dict[dict_key]["example_label_for_each_img"]
        y_unique_list  = y.unique().tolist()
        class_colors   = pca_data_dict[dict_key]["example_class_labels_colors_dict"] # y_unique is a key, color is a value

        # plot each class separately, 
        for one_y_unique in y_unique_list:
            'one_y_unique - may have only one class, or all from color dict'
            
            # find items, and select color for points
            oyu_idx = y==one_y_unique
            one_class_color = class_colors[one_y_unique]
    
            if adjust_markeralpha==False:
                final_markeralpha = markeralpha
            else:
                '''
                    alpha is proportional to percentage of class 
                    in total population in a given dataset == then rarerr the point is, 
                    then higher alpha it will get 
                '''
                adjusted_alpha = 1-(np.array(y==one_y_unique).sum()/len(y))
                if adjusted_alpha<min_adjusted_markeralpha:
                    adjusted_alpha=min_adjusted_markeralpha
                else:
                    pass
                final_markeralpha = adjusted_alpha
 
            # Plot their components
            ax.scatter( 
                X_pca[oyu_idx, 0],                 # if more axes woudl be used, 
                X_pca[oyu_idx, 1],                 #. then X_2d woudl have more collumns, 
                label   =one_y_unique,
                s       =markersize,
                color   =one_class_color,
                alpha   =final_markeralpha
                )
        # ... Labels and legend
        ax.set_title(dict_key)
        ax.set_xlabel('1st component - principial axes')
        ax.set_ylabel('2nd component')
        handles, labels = ax.get_legend_handles_labels()
        # ...
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        # ...
        ax.spines["right"].set_visible(False) # and below, remove white border, 
        ax.spines["top"].set_visible(False)

        # ensure that ax box is square
        #ax.set_aspect('equal', 'box')

        # Legend, ........................................
        l = ax.legend(handles, labels, loc=(1, 0), fontsize=10, frameon=False) # bbox_to_anchor = (0,-0.1,1,1)
        l.set_title('Class:',prop={'size':10}) # otherwise, I can not control fontsize of the title,

    # layout, 
    fig.tight_layout()
    fig.subplots_adjust(top=subplots_adjust_top)
    plt.show();
    
    
    

# Function, ..........................................................................................
# new 2020.12.12
def new_pca_screeplot(*,   
    pca_data_dict,
    title=None,
    figsize_per_subplot=(5,4),
    ylim_max=0.11,
    x_overhangs=20,   
    title_fontsize=25, 
    subplots_adjust_top=0.7,
    annotate_lines=True
):
    '''
        Function that creates scree plot for pca reults stored in dictionary, 
        each subplot, has two axes, one with proportion of varinace explained by each principial components, shown as barplot,
        and second y axis, with different scale (0-100%) showing cumulatice pve as % of explained vaiance, using stepplot, 
        Caution: you must manually sset number of columns and rows on the figure for setting subplot nr right, 
    
            # Input,
            . pca_models_dict: dictionary, with trained and fitted pca models, 
                               dict.key will be used as subplot title, 
            . ....
            . title         : str, 
            . ylim_max      : between 0, and 1.1
            . x_overhangs   : int, how much the x axis shodul be longer on both sites arround the plot, 
                              to allow visualising all the lines, 
                              from each site it will be xmin-x_overhangs/2, and xmax+x_overhangs/2
            . ....
            . figsize_per_subplot=(5,4) - first digit is mutiplied by the number of subplots
                                        - second int, is a figure heigh
            . annotate_lines: bool, is True, guidlines, showing the number of princial axes that allow explanining 
                              10%, 20% 50% 90% and 99% or 100%  will be annotated with text, 
                
            # returns, 
            . matplotlib figure, 
    '''            
                
                  
    # Params, 
    fig_ncols = len(pca_data_dict)
    fig_nrows = 1
    figsize   = (figsize_per_subplot[0]*fig_ncols, figsize_per_subplot[1])
    # .....
    bar_color = "blue"
    step_plot_linecolor = "red"
    gridline_linecolor = "grey"
    
    
    # Make figure, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(nrows=fig_nrows, ncols=fig_ncols, facecolor="white", figsize=figsize)
    if title!=None:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        pass
      
    # ... in case you have only one subplot, 
    if fig_ncols*fig_nrows==1:
        axss = [axs]
    else:
        axss = axs.flat  

    # Load, transform, and plot dataset create with each module, 
    for i, ax in enumerate(axss):
        
        # - 0 - Data preparation,
        # .... title, 
        'used in - 2 -'
        dict_key = list(pca_data_dict.keys())[i]
        subplot_title = dict_key
        
        # .... extract the % of the variance represented by each principial compenent 
        pve = pca_data_dict[dict_key]["pca__saved_model"].explained_variance_ratio_

        # .... calculate cumulative sum
        pve_cumsum = np.cumsum(pve)

        # .... find how many components explain 10%, 20%, 90% and 100% of the varinace in the features ?
        # pve_th = np.linspace(0.1, 1, 10, endpoint=True) # - these will be displayed as lines, 
        pve_th = np.array([0.1, 0.2, 0.5, 0.9, 1])
        pve_th_feature_nr = list()
        for j, p in enumerate(pve_th):
            try:
                pve_th_feature_nr.append(np.where(pve_cumsum>=p)[0][0])
            except:
                # if not availbel, take the last one, 
                pve_cumsum_idx = np.where(pve_cumsum<p)[0][-1]
                pve_th_feature_nr.append(pve_cumsum_idx)
                # .. and update, corresponding pve_th
                pve_th[j]= pve_cumsum[pve_cumsum_idx]


            
            
        # - 1 - baplot on scree-plot, 
        '''
            small barplot with proportion of explaned variance with each feature 
            - on the bottom, bearly visible with large number of features
        '''
        # ..... barplot, with varinace explained by each p. component
        xcordinates = np.arange(1, len(pve) + 1) # 1,2,..,n_components
        ax.bar(xcordinates, pve, color=bar_color, edgecolor=bar_color)  
        
        # ..... aestetics, 
        ax.set_ylim(0, ylim_max) 
        ax.set_xlim(-x_overhangs/2, len(pve)+x_overhangs/2)
        ax.spines["right"].set_visible(False) # and below, remove white border, 
        ax.spines["top"].set_visible(False)
        ax.set(
                title=list(pca_data_dict.keys())[i],
                xlabel="Principial component number")
        ax.set_ylabel('proportion of\nvariance explained(pve)')

 
        # - 2 - step plot on ax2, 
        '''
            step plot with cumulative sum of explanie variances
            located on the same lot as barplot, 
        '''
        # ..... instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx() 
        ax2.set_ylim(0,1.1)

        # ..... step line with cumulative sum
        ax2.step(
            xcordinates+0.5, # 1.5,2.5,..,n_components+0.5
            pve_cumsum, # Cumulative sum
            color=step_plot_linecolor,
            label='cumulative pve',
            lw=0.5
        )

        # ..... title ad aestetists, 
        ax2.set( xlabel="Principial components")
        ax2.set_ylabel('cumulative percentage\nof explained variance', color="red")
        ax2.tick_params(axis='y')
        ax2.spines["top"].set_visible(False)
        # legend
        #ax2.legend(frameon=True, loc="center right")

        # ..... set yticks, for ax2
        ax2.set_yticks(np.linspace(0.1, 1, 10, endpoint=True))
        ax2.set_yticklabels(   
            [ f"{str(int(x*100))}%" for x in np.linspace(0.1, 1, 10, endpoint=True).tolist()],
            color=step_plot_linecolor
        )

        
        # - 3 - add guidlineslines, 
        '''
            gridlines help estimating how many principial components 
            allow explaning 10%, 20% etc.. of varinace in data
        '''
        gridline_linecolor = "grey"

        # add vertical lines, 
        for j, xline in enumerate(pve_th_feature_nr):
            ymax = pve_cumsum[xline]*10/11 # multiplied by 10/11 because ymax in axhline is on axis cordinates from 0 to 1.
            ax2.axvline(xline, ymin=0, ymax=ymax, lw=0.5, color=gridline_linecolor, ls=":")

        # add horizontal lines, 
        for j,(p, xline) in enumerate(zip(pve_th, pve_th_feature_nr)):
            plot_x_size = (len(pve)+20)
            xmin = xline/plot_x_size
            ax2.axhline(p, xmin=xmin, xmax=plot_x_size, lw=0.5, color=gridline_linecolor, ls=":")


        # add annotastion, with number of pca axes that allow explainign certain % of varinace, 
        if annotate_lines==True:
            for j,(p, xline) in enumerate(zip(pve_th, pve_th_feature_nr)):
                if (xline+1)==1:
                    annotation_end="p.component"
                else:
                    annotation_end="p.components"
                line_annotation_text = f"{int(p*100)}% explained with {xline+1} {annotation_end}"
                # ...
                text_x_position = len(pve)*0.18
                text_y_position = (pve_cumsum[xline])
                ax2.text(text_x_position, text_y_position, line_annotation_text, fontsize=8, color=gridline_linecolor, ha="left")
            else:
                pass
                     
    plt.tight_layout()    
    plt.subplots_adjust(top=subplots_adjust_top)
    plt.show();
    
    
    

    

    
# Function, ..........................................................................................
# new 2020.12.12    
def pve_vs_featurevariance_sizescatterplots(*, 
    pca_data_dict,
    pca_axes_max_nr=10,            
    title=None,
    fig_nrows=1, 
    fig_ncols=1,
    figsize_per_subplot=(5,4),
    upper_ylim=0.2, 
    scale_for_markers=20,
    add_noise=False, 
    xlimits=None, 
    title_fontsize=25, 
    subplots_adjust_top=0.7
):
    
    # Params, 
    fig_ncols = len(pca_data_dict)
    fig_nrows = 1
    figsize = (figsize_per_subplot[0]*fig_ncols, figsize_per_subplot[1])
    # .....
    markercolor="steelblue"
    markeralpha=0.2
 
    # dict to store results,
    rdct = dict()

    # Make figure, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(nrows=fig_nrows, ncols=fig_ncols, facecolor="white", figsize=figsize)
    if title!=None:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        pass
      
    # axes must but a list - in case you have only one subplot, 
    if fig_ncols*fig_nrows==1:
        axss = [axs]
    else:
        axss = axs.flat  

    # Load, transform, and each dataset in pca_data_dict 
    for i, ax in enumerate(axss):

        # - 1 - get data for the subplot, 
        dict_key       = list(pca_data_dict.keys())[i]
        subplot_title  = dict_key
        X              = pca_data_dict[dict_key]["example_extracted_features"]
        y              = pca_data_dict[dict_key]["example_label_for_each_img"]
        y_unique_list  = y.unique().tolist()
        class_colors   = pca_data_dict[dict_key]["example_class_labels_colors_dict"] # y_unique is a key, color is a value
        pca_model      = pca_data_dict[dict_key]["pca__saved_model"]

        
        
        # - 2 - data preparation
        # ..... select how many principial components to check
        if pca_axes_max_nr==None:
            idx_max = X.shape[0]
        else:
            idx_max = pca_axes_max_nr

        # ..... x-axis, (array where rows: principial components, cols: features,)
        '''
            array where rows: principial components, cols: features
            cols are: 1:pve, 2:variance in the feature, 3:feature corr with paxis
        '''    
        pca_components_corr_with_features = pca_model.components_[0:idx_max,:]
        pve                               = pca_model.explained_variance_ratio_[0:idx_max]
        feature_var                       = X.var(axis=0)
        
        # ..... from time to time, standard scaller generate very small negative values, thus:
        feature_var = feature_var.clip(0) # these will be turned to zero, the issue was described in internet, 
            
        
        # ..... prepare array_for_plot
        for row_counter, pve_i in enumerate(pve.flatten().tolist()):
            pve_x_features = np.array([pve_i]*feature_var.shape[0])
            temp_arr = np.c_[pve_x_features, feature_var, pca_components_corr_with_features[row_counter,:].flatten()]
            # ...
            if add_noise==True:
                # adding small gausian noise to varinace, to better visualize the points, 
                #.       - done in order to allow seeing points in standarized datsets
                random_noise_to_x = np.random.rand(temp_arr.shape[0])*np.random.choice(np.array([-1,1]), size=temp_arr.shape[0], replace=True)/100
                random_noise_to_y = np.random.rand(temp_arr.shape[0])*np.random.choice(np.array([-1,1]), size=temp_arr.shape[0], replace=True)/100
                temp_arr[:,1] = temp_arr[:,1]+random_noise_to_x
                temp_arr[:,0] = temp_arr[:,0]+random_noise_to_y
            else:
                pass
            # ...
            if row_counter==0:
                array_for_plot = temp_arr
            else:
                array_for_plot = np.r_[array_for_plot, temp_arr]

             
            
        # - 3 - make plot            
        ax.scatter(
            array_for_plot[:,1],    # x - feature variance, 
            array_for_plot[:,0],    # y - pve on axis, 
            s=((np.absolute(array_for_plot[:,2]))*scale_for_markers)**2, # correlation between principial component and a given feature,
            color=markercolor, 
            alpha=markeralpha,
            marker="o"
        )
            
        # ..... auestetics
        ax.set_ylim(0, upper_ylim)
        ax.spines["right"].set_visible(False) # and below, remove white border, 
        ax.spines["top"].set_visible(False)            
        if xlimits==None:
            pass
        else:
            ax.set(xlim=xlimits)
        
        # ..... title & labels,
        ax.set_title(subplot_title) 
        ax.set_xlabel("Variance in each feture")
        ax.set_ylabel('proportion of\nvariance explained(pve)')
 
        # ..... legend
        legend_point_size = [1, 0.5, 0.2, 0.1]
        for lps in legend_point_size:
            ax.scatter([], [], c="red", alpha=markeralpha, s=(np.absolute(lps)*scale_for_markers)**2, label=f"abs(r)={lps}")
        l = ax.legend(scatterpoints=1, frameon=False, labelspacing=1.5, fontsize=8, loc=(1,0.1))
        l.set_title("Correlation (r) between\nprincipial component\nand a given feature",prop={'size':8})
 

    # figure settings, 
    plt.tight_layout()    
    plt.subplots_adjust(top=subplots_adjust_top)
    plt.show();




    
    
    