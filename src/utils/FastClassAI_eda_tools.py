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

from src.utils.data_loaders import load_encoded_imgbatch_using_logfile
from src.utils.data_loaders import load_raw_img_batch
from src.utils.data_loaders import load_raw_img_batch_and_batch_labels
from src.utils.clustered_histogram import add_descriptive_notes_to_each_cluster_in_batch_labels
from src.utils.clustered_histogram import find_clusters_on_dendrogram
from src.utils.clustered_histogram import create_clustered_heatmap_with_img_examples
from src.utils.annotated_pie_charts import annotated_pie_chart_with_class_and_group 
from src.utils.annotated_pie_charts import prepare_img_classname_and_groupname
from src.utils.tools_for_plots import create_class_colors_dict
from src.utils.example_plots_after_clustering import plot_img_examples
from src.utils.example_plots_after_clustering import create_spaces_between_img_clusters
from src.utils.example_plots_after_clustering import plot_img_examples_from_dendrogram
from src.utils.pca_tools import pca_analyis_with_plots
from src.utils.pca_tools import pca_scree_plots
from src.utils.pca_tools import pca_pve_vs_variance_scatterplots
from src.utils.annotated_pie_charts import annotated_pie_chart_with_class_and_group
from src.utils.annotated_pie_charts import prepare_img_classname_and_groupname





# function ...................................................................................
def cluster_images_and_plot_examples(*,
        path,
        batch_names,
        dataset_name,
        dataset_variants,
        module_names,
        class_labels_configs,
        max_pie_nr = 7,
        n_img_examples  =100 ,
        max_img_per_col = 10,
        figsize_scaling = 1.5
):
    '''
        Function that creates 4 plots
        1. plot based on seaborn clustered histogram, with histogram showing feaures extracted with a given module
        2. plot with annotated pie charts shwoing which clusters distinguishen on the histogram are in each lalleled class
        3. same as above, but here it shows how different classes are placed in different clusters on the histogram
        4. larger number of image examples from the debdrogram on a first plot
        
        comment: this is a custom funciton created to work with batches of raw images
                 and features extracted from these batches using FastClassAI pipeline
        
        params
        ..........................................................
        batch_names          : list, with strings, in 
        dataset_name         : str, dataset name, used to create file amd folder names
        dataset_variants     : list, with str's , used to create file amd folder names
        module_names         : list, with str, with modules names, used to create file amd folder names
        n_img_examples       : number of image examples on plot 4. 
        class_labels_configs : dict with dict, 
                               fist level dict: key == dataset_variant
                               second level dict: key == class_labels_colors
        max_pie_nr           : max nr of pie charts displayed in one row on Plot No 2 and 3
        n_img_examples       : int, number of images displayed on plot No 4, if less valailable , 
                               only max number will be displayed 
        max_img_per_col      : int, max nr of images displayed in one column on plot No 4
        figsize_scaling      : float, used to scale images on plot No 4                 
        
        
        returns
        ..........................................................
        4 plots decribed in the above, 
    '''

    # ....................................................
    for dataset_variant in dataset_variants:

        # find colors assigned to each class, 
        class_colors  = class_labels_configs[dataset_variant]['class_labels_colors'] # speciffic for dataset variant

        # paths
        '''extracted features, and labels are loaded using logfiles created while extracting features'''
        path_raw_images_and_log_files = os.path.join(path, f"{dataset_name}__{dataset_variant}")
        path_extracted_features       = os.path.join(path, f"{dataset_name}__{dataset_variant}__extracted_features")

        
        # ...............................................
        for module_name in module_names:

            
            # -- loading data for plot No1 & 4 --
            
            # Load raw img,  
            '''
               Extracted features, and labels are loaded using logfiles created while extracting features
               I am loadiung these images at each time becuase cv2 library, used with the program 
               was often creating changes in them while adding text annotaitons
               moreover it is relatively fast to load just one or few batches of images, 
            '''
            raw_img_batch = load_raw_img_batch(
                load_datasetnames    = batch_names, 
                path                 = path_raw_images_and_log_files, 
                image_size           = (500,500), 
                verbose              = False
            )

            
            # -- plot No1 --
            
            # Plot examples from each class, and heatmap with clustered features and images,
            'function returns dict with results for a given module'
            clustering_results = create_clustered_heatmap_with_img_examples(
                raw_img_batch                            = raw_img_batch,
                load_dir                                 = None, 
                extracted_features_load_dir              = path_extracted_features,
                logfiles_for_extracted_features_load_dir = path_extracted_features,  # usually, the same as extracted_features_load_dir      
                dataset_name                             = dataset_name,
                module_names                             = [module_name], # if more modules are proivided, data from each will be used sequencially, and results sotred in returned dict
                subset_names                             = batch_names, # list, with at least one data set name,
                class_colors                             = class_colors
            )
            plt.show();

            
            # -- preparing data for plots No2 & 3 --
            
            # prepare lists with classnames and clusternames for annotated pie charts, 
            img_classnames, img_groupnames = prepare_img_classname_and_groupname(
                data_for_plot          = clustering_results[module_name],
                number_of_img_examples = 65000 # it will use max available images, 
                )

            # .. create distionary with clusternames and assigned colors, 
            class_color_for_clusters = create_class_colors_dict(
                list_of_unique_names = pd.Series(img_groupnames).unique().tolist(),
                cmap_name="Purples", cmap_colors_from = 0.5, cmap_colors_to = 1
                )
       
            
            # -- plot No2 --
            
            # .. Pie chart set 1, 
            annotated_pie_chart_with_class_and_group(
                title="Class Assigment to different clusters on dendrogram",
                classnames=img_groupnames, 
                groupnames=img_classnames, 
                groupname_colors=class_colors, 
                n_subplots_in_row=max_pie_nr, 
                class_colors=class_color_for_clusters, 
                legend_loc="upper right"
            )

      
            # -- plot No3 --
            
            # .. Pie chart set 2,
            annotated_pie_chart_with_class_and_group(
                title="Cluster Composition",
                classnames=img_classnames, 
                groupnames=img_groupnames, 
                class_colors = class_colors, 
                n_subplots_in_row=max_pie_nr, 
                mid_pie_circle_color="lightblue", 
                legend_loc="upper right"
            )
            
            
            # -- plot No4 --
            
            # Plot Image Examples from each cluster,  
            plot_img_examples_from_dendrogram(
                number_of_img_examples = n_img_examples,
                raw_img_batch          = raw_img_batch,
                data_for_plot          = clustering_results[module_name],
                plot_title             = f"{n_img_examples} examples from {dataset_variant}, {module_name}, ordered as on dendrogram",
                max_img_per_col        = max_img_per_col,
                figsize_scaling        = figsize_scaling
            )
            
            
    
    
    

# function ...................................................................................
def pca_analysis_on_plots(*,
        path,
        batch_names,
        dataset_name,
        dataset_variants,
        module_names,
        class_labels_configs,
        pca_figsize     = (12,4), # tuple (row lengh, row height)
        pca_axes_max_nr = 250, 
        verbose         = False,
        scale_data      = False
):

    '''
       Function that perform PCA analisis on features sextracted from one or more batches of images, 
       the results are presented as plots with subplots presenting results from different modules, 
       Plots:
       No 1. PCA plot, 2 first componenets
       No 2. Scree plot with dot lines showing how many principial axes may explain 10%, 20% ...99% of the varinace in the dataste
       No 3. plot showing correlations between each features and each principial axis, used to check 
             if there are not features wiht high varinace that are strongly correlated with snall number of pca axes
             is yes, try to repeat the analyis using scaling
        
        params
        ..........................................................
        batch_names          : list, with strings, in 
        dataset_name         : str, dataset name, used to create file amd folder names
        dataset_variants     : list, with str's , used to create file amd folder names
        module_names         : list, with str, with modules names, used to create file amd folder names
        class_labels_configs : dict with dict, 
                               fist level dict: key == dataset_variant
                               second level dict: key == class_labels_colors
        pca_figsize          : tuple, with int's (row lengh, row height)
        pca_axes_max_nr      : int, nr of pca features analized on plot No3 
        verbose              : bool,
        scale_data           : bool, if True, features are scaled (mean=0, sd=1)        
        
        returns
        ..........................................................
        3 plots decribed in the above, 
    
    '''
  
  
  
  
    for dataset_variant in dataset_variants:

    
        mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings,
    
        # find colors assigned to each class, 
        class_colors  = class_labels_configs[dataset_variant]['class_labels_colors'] # speciffic for dataset variant

        # paths
        '''extracted features, and labels are loaded using logfiles created while extracting features'''
        path_raw_images_and_log_files = os.path.join(path, f"{dataset_name}__{dataset_variant}")
        path_extracted_features       = os.path.join(path, f"{dataset_name}__{dataset_variant}__extracted_features")

        # pca plot and analysis for the two remaining plots, 
        pca_models_dict = pca_analyis_with_plots(
                title=f"{dataset_name}, {dataset_variant}\nPCA: visualization of extracted features",
                # ...
                path=path_extracted_features,
                verbose=False,
                dataset_name=dataset_name,
                module_names=module_names,
                subset_names=batch_names,
                class_colors=class_colors,
                # ...
                scale_data=scale_data,
                # ..
                fig_nrows=1,
                fig_ncols=len(module_names),
                figsize=pca_figsize,
                title_fontsize=12,
                subplots_adjust_top=0.8
            )

        pca_scree_plots(
                title=f"{dataset_name}, {dataset_variant}\nScree plot",
                pca_models_dict=pca_models_dict, 
                ylim_max=0.2,
                fig_ncols=len(module_names),
                figsize=pca_figsize,
                title_fontsize=12,
                subplots_adjust_top=0.8
            )

        _ = pca_pve_vs_variance_scatterplots(
                title=f"{dataset_name}, {dataset_variant}\nCorrelation between each feature\nand each principial component",
                pca_models_dict=pca_models_dict, 
                path=path_extracted_features, 
                # ...
                dataset_name=dataset_name, 
                subset_names=batch_names, 
                module_names=module_names,
                # ...
                fig_ncols=len(module_names),
                figsize=pca_figsize,
                title_fontsize=12,
                subplots_adjust_top=0.8,
                # ...
                pca_axes_max_nr=pca_axes_max_nr, # 250 first pricial components will be used to make the plot, 
                scale_data=scale_data,
                add_noise=scale_data # add small gausian noise to allow better visualisation of overlapping data points
            )
        
        
        
        
     
    
    
    
    
        
# function ...................................................................................
def plot_n_image_examples(*,   
        n = 100,             # int, max number of displayed images,
        plot_title=None,     # If None, you must provide dataset_name & dataset_variant
        use_fastclassai_labels = False, 
        
        # paths,                   
        path,                # eg PATH_interim, where you store both , used with pipelines cretead with FastClassAI
        raw_img_path=None,   # in case raw images are stored somewhere else
        batch_labels_path=None, # also in case batch labels are stored somewhere else
        
        # data and names, 
        dataset_name,        # required if use_fastclassai_labels==True, for FastClassAI path and plot naming
        dataset_variant,     # required if use_fastclassai_labels==True, for FastClassAI path and plot naming
        module_name,         # required if use_fastclassai_labels==True, any module used for feature extraction is ok
        batch_name,          # name of one batch, or list with name sof several batches,
        class_labels_colors=None, # dict {key: classname, value: color}
        
        # keras image generator params, 
        img_size =(500, 500),# size of loaded images, 
        batch_size=1000,     # no more then 1000 images will be loaded, CAUTIOn, you may load more images, and then use n to display less, but with examples from all 
        shuffle_images=False,# bool, if True, images in the batch will be randomly shuflled, but later on ordered into classesm f these are available, You shoudl use that if you have more images loaded examples,   
        
        # plot eastetics                  
        figsize_scaling=1,     # float, affected size of all items, from fonts to images, I did my best to make it work at many different scales                
        font_scaling = 1,      # float, affects size of title and lenged font, 
        legend_pos_to_left =0, # float, fraction of the image total size, will be added to horozontal position of the center of the legend, 
        max_img_per_col=None,   # int, or None, how many images will be displayed max, on each column, then less, then wider is the image, None is automatic                       
        class_colors_cmap = "tab20" # str,  used to create unique colors for each class, not used if class_labels_colors is provided 
                         
):
    '''
        Displays image examples, grouped by class, from one or more natches of images, 
        Two options are availabel to load images and batches,
        1. you use only image keras generators, to get both from images sorted into class-names folders
        2. you may load images as in the above, but to labels are loaded from 
        batch labels csv files created for each batch, while extracting features wiht tf-hub module
        with FastClassAI pipeline, 
        
        Caution: 
        * From time to time, I had some artifact visible on the image, like a line without the color, 
          to remove it just chnage figsize_scaling a little, eg use 1.1 instead of 1, 
          
        # params
        . n                          : int, max number of displayed images,
        . plot_title                 : str, If None, you must provide dataset_name & dataset_variant
        
        # paths,             
        . use_fastclassai_labels     : bool,  if yes, you will use FastClassAI pieline path, and you need to provide 
                                       module_name, dataset_name, and dataset_variant, and make sure that features 
                                       were extracted already from images
        . path,                      : eg PATH_interim, where you store both , used with pipelines cretead with FastClassAI
        . raw_img_path=None,         : in case raw images are stored somewhere else
        . batch_labels_path=None,    : also in case batch labels are stored somewhere else
        
        # data and names, 
        . dataset_name,              : str, required if use_fastclassai_labels==True, for FastClassAI path and plot naming
        . dataset_variant,           : str, required if use_fastclassai_labels==True, for FastClassAI path and plot naming
        . module_name,               : str, required if use_fastclassai_labels==True, any module used for feature extraction is ok
        . batch_name,                : str, or list, name of one batch, or list with name sof several batches,
        . class_labels_colors=None,  : dict {key: classname, value: color}
        
        # keras image generator params, 
        . img_size                   : tuple, size of loaded images in pixels,  
        . batch_size                 : int, eg 1000, no more then 1000 images will be loaded, 
                                       CAUTIOn, you may load more images, and then use n to display less, 
                                       but with examples from all 
        . shuffle_images             : bool, if True, images in the batch will be randomly shuflled, 
                                       but later on ordered into classes of these are available, 
                                       You shoudl use that if you have more images loaded examples,   
        # plot eastetics                  
        . figsize_scaling            : float, affected size of all items, from fonts to images, 
                                       I did my best to make it work at many different scales                
        . font_scaling               : float, affects size of title and lenged font, 
        . legend_pos_to_left         : float, fraction of the image total size, will be added 
                                       to horozontal position of the center of the legend, 
        . max_img_per_col            : int, or None, how many images will be displayed max, 
                                       on each column, then less, then wider is the image, None is automatic                       
        . class_colors_cmap          : str,  used to create unique colors for each class, not used 
                                       if class_labels_colors is provided    
    '''
    
    
    # ** / PATHS AND LISTS CORRECTIONS / **

    # batch_name must be a list for load_raw_img_batch
    if isinstance(batch_name, list): 
        batch_name_list = batch_name
    else: 
        batch_name_list = [batch_name]
        
    # paths in FastClassAI pipeline
    if raw_img_path==None:
        raw_img_path = os.path.join(path, f"{dataset_name}__{dataset_variant}")
    else:
        pass        
        
    # path in FastClassAI pipeline    
    if batch_labels_path==None:
        batch_labels_path=os.path.join(path, f"{dataset_name}__{dataset_variant}__extracted_features")
    else:
        pass
    
    
    
    # ** / LOAD IMAGES AND LABELS / **
    '''
        two options are availabel to load images and batches,
        1. you use only image keras generators, to get both from images sorted into class-names folders
        2. you may load images as in the above, but to labels are loaded from 
        batch labels csv files created for each batch, while extracting features wiht tf-hub module
        with FastClassAI pipeline, 
    '''    
        
    # - OPTION 1. 
    #.  load images and batch labels directly from sorted images using 
    if use_fastclassai_labels==False:
        raw_img_batch, _, _, batch_labels = load_raw_img_batch_and_batch_labels(
            load_datasetnames = batch_name_list, 
            path = raw_img_path,         
            image_size=img_size, 
            batch_size=batch_size,
            shuffle_images=shuffle_images,
            verbose=False
        )
        # randomly select unique number of requested images that are evenly spaced across entire batch
        img_idx = np.unique(np.floor(np.linspace(0,raw_img_batch.shape[0], n, endpoint=False)).astype(int)).tolist()

        # make pd series
        batch_labels = pd.Series(batch_labels)
        
           
    # - OPTION 2. 
    #.  load images and batch labels using batch labels create 
    else:

        # load example of raw imgages, 
        raw_img_batch = load_raw_img_batch(
            load_datasetnames=batch_name_list, # if more batch names provided, they will be joinned together
            path=raw_img_path,
            image_size=img_size, 
            batch_size=batch_size,
            shuffle_images=shuffle_images,
            verbose=False
        )

        # randomly select unique number of requested images that are evenly spaced across entire batch
        img_idx = np.unique(np.floor(np.linspace(0,raw_img_batch.shape[0], n, endpoint=False)).astype(int)).tolist()

        # find logfile and later on batch labels for loaded images
        'shodul be only one logfile for a combinations of dataset varinat and tf module'
        os.chdir(batch_labels_path)
        for i, file in enumerate(glob.glob(f"{module_name}*{dataset_variant}*logfile*")):
            if i==0: logfile = pd.read_csv(file) 
            else: pass

        # load batch labels - to have a column with class name for each image, 
        'caution there could be >1'
        for i, bn in enumerate(batch_name_list):
            if i==0:
                batch_labels = pd.read_csv(logfile.img_batch_info_filename.loc[logfile.datasetname==bn,].values[0])
            else:
                batch_labels_next = pd.read_csv(logfile.img_batch_info_filename.loc[logfile.datasetname==bn,].values[0])
                batch_labels = pd.concat([batch_labels, batch_labels_next], axis=0)
                batch_labels.reset_index(inplace=True)

        # extract only batch labels, from the df, 
        batch_labels = pd.Series(batch_labels.classname)        
    
    
    # ** / FIND COLORS FOR EACH IMAGE / **
    
    # class colors, - chnages because I added legend that looks really nice, 
    if class_labels_colors==None:
        class_labels_colors = create_class_colors_dict(
            list_of_unique_names = pd.Series(batch_labels).unique().tolist(),
            cmap_name=class_colors_cmap, 
            cmap_colors_from = 0, 
            cmap_colors_to = 1
            )
    else:
        pass    


    
    # ** / PLOT IMAGES / **
    
    # map class clabel colors to classnames in batch labes
    color_for_each_img        = pd.Series(batch_labels.map(class_labels_colors))
    
    # title
    if plot_title == None:
        try:
            if dataset_name!=None or dataset_variant!=None:
                plot_title = f"{dataset_name}, {dataset_variant}, {len(img_idx)} images"
            else:
                plot_title = ""
        except:
            plot_title = ""
    else:
        pass
    
    # plot exaples, 
    plot_img_examples(
        title                         = f"{dataset_name}, {dataset_variant}, {len(img_idx)} images",
        selected_img_batch            = raw_img_batch[img_idx], 
        img_groupname                 = (batch_labels.iloc[img_idx]).values.tolist(), 
        img_color                     = (color_for_each_img.iloc[img_idx]).values.tolist(),
        class_colors_for_legend       = class_labels_colors,  # dictionary taken from class_labels_configs

        # cgeneral settings 
        figsize_scaling                 = figsize_scaling,
        subplots_adjust_top             = 0.7,        
        display_groupname               = False,
        max_img_per_col                 = max_img_per_col,
        display_img_number_on_each_image= False,
        
        # fonts
        title_fontsize_scale           = 1.6*font_scaling,
        legend_fontsize_scale          = 2.2*font_scaling, 
        
        # legend
        legend_loc                     = "center",
        legend_ncol                    = 3,
        legend_postions_vertical_add   = -0.05,
        legend_postions_horizontal_add = -0.1+legend_pos_to_left
        
    )    
        
      
      
      
      

  
  
  
# Function, ..................................................................
def annotated_piecharts_with_subset_class_composition(*, 
    # data and title
    plot_title=None,
    subset_names,
    displayed_subset_names=None, 
    class_labels_colors=None,
    plot_class_composition=True,

    # required to find logfiles, 
    path, 
    dataset_name,
    dataset_variant,
    module_name,

    # fig aestetics, 
    figsize_scale          = 1.8,
    figheight_scale        = 2,    
    figwidth_scale         = 1,
    adjust_top             = 0.95,
                                                      
    fontsize_scale         = 1,
    title_fontsize_scale   = 1,
    legend_loc             = (0.1,0.7),
    legend_ncol            = 4,
    n_subplots_in_row      = 3,     
                                                      
    # class colors, 
    class_colors_cmap      = "Greens",
    cmap_colors_from       = 0.4, # so it doent starts with white color!
    cmap_colors_to         = 1
):    
    '''
        FastClassAI pipeline function, creates annotated pie charts showing class and file/image number and %
        in each class in different subsets, subsets are displayed as different subplots, each with pie chart
        class instances in each subset are displayed as pie chart franctions with different colors, 
        
        This function uses logfiles created by feature extraction with tf hub module, 
        to find batch labels files. This function can not work without these files, 
        but it is based on lower level annotated_pie_chart_with_class_and_group() 
        function that can be easily used separately for any set of data, with mutiple 
        classes and goups to create the same plots.
        
        # data and title
        . plot_title             : str, or None, 
        . subset_names           : str, or list, eg: ["train", "valid", "test"], 
                                   name of one batch, or list with name of several batches, 
                                   that will be displayed as separate subplots. Caution, these names 
                                   are used to search pattern and to identify unambigously set of batch files,     
        . displayed_subset_names : None, str or list of the same lenght as subset_names
                                   provide your names that are nicer to read, if None subset_names 
                                   will be used
        . class_labels_colors    : dict, key = classname, value = color
        . plot_class_composition : bool, if True, it will plot class composition, if False, it will create only one pie chart 
                                   with nr and % of files/images in each subset

        # required to find logfiles, 
        . path                   : str, eg PATH_interim, where you store both  
        . dataset_name           : required, for path and naming
        . dataset_variant        : required, for path and naming
        . module_name            : any module used for feature extraction is ok

        # fig aestetics, 
        . figsize_scale          : float, eg: 1.8 affects scale of pie charts 
                                   in comparison to the rest of the subplot
        . fontsize_scale         : float, default=1
        . legend_loc             : tuple, default=(0.1,0.7), 
        . legend_ncol            : int, legend ncol parameter,
        . n_subplots_in_row      : int, how many subplots
        
        . class_colors_cmap      : str, matplotlib cmap name, used only, 
                                   if class_labels_colors==None, 
    
    '''
    
    
    # if None, use FastClassAI convention, 
    if plot_title==None:
        plot_title=f'{dataset_name}, {dataset_variant}'
    else:
        pass  

    # you need a list,m but you may have a string that must be converted, 
    if isinstance(subset_names, list):
        pass
    else:
        subset_names = [subset_names]

    # you need a list,m but you may have a string that must be converted, 
    if displayed_subset_names!=None:
        if isinstance(displayed_subset_names, list):
            pass
        else:
            displayed_subset_names = [displayed_subset_names]        
    else:
        displayed_subset_names=subset_names

    # add what is the percentage of images/files displayed on each subplot, (adds to 100% on all subplots)
    if len(subset_names)==1 or plot_class_composition==False:
         add_group_item_perc_to_numbers_in_each_pie=False
    else:
         add_group_item_perc_to_numbers_in_each_pie=True
            
            
            
        
    # / plot each subset separately /     

    # go to dir with file labels created when extracting (FastClassAI pipeline)
    os.chdir(os.path.join(path, f"{dataset_name}__{dataset_variant}__extracted_features"))

    # collect the data for each subset
    for sn_i, (one_displayed_subset_name, one_subset_name) in enumerate(zip(displayed_subset_names,subset_names)):   
        # load and concatenate all fil labels associated wiht a train, test, or valid subsets
        for i, file in enumerate(glob.glob(f"{module_name}*{one_subset_name}*labels.csv")):
            if i==0:
                labels_df = pd.read_csv(file)
            else:
                labels_df = pd.concat([labels_df, pd.read_csv(file)], axis=0)

        # collect classname and subset name
        if sn_i==0:
            class_labels  = pd.Series(labels_df.classname)
            subset_labels = pd.Series([one_displayed_subset_name]*class_labels.shape[0])
        else:
            another_class_labels  = pd.Series(labels_df.classname)
            another_subset_labels = pd.Series([one_displayed_subset_name]*another_class_labels.shape[0])        
            # join
            class_labels  = pd.concat([class_labels, another_class_labels])
            subset_labels  = pd.concat([subset_labels, another_subset_labels])
            # index reset
            class_labels.reset_index(inplace=True, drop=True)
            subset_labels.reset_index(inplace=True, drop=True)

            
            
    # adapt data for two types of displayed info:
    if  plot_class_composition==True:
        '''
            default plots
        '''
        classnames_values             = class_labels.values.tolist() # different classes displayed as different colors opn one pie chart
        groupnames_values             = subset_labels.values.tolist() # different groupenames == different subplots , each with pie chart
        class_colors_values           = class_labels_colors # used for assigning colors on pie charts to classnames, assigned alphabetically to names on 0-1 cmap scale
        """
        if pd.Series(classnames_values).unique().shape[0]==1:
            groupnames_values             = ["file nr:"]*len(classnames_values)
        else:
            pass        
        
        """
    else:
        classnames_values             = subset_labels.values.tolist() # different groupenames == different subplots , each with pie chart
        groupnames_values             = ["file nr:"]*len(classnames_values)
        class_colors_values           = None
            
            
    # plot
    annotated_pie_chart_with_class_and_group(

                    classnames             = classnames_values, # different classes displayed as different colors opn one pie chart
                    groupnames             = groupnames_values, # different groupenames == different subplots , each with pie chart
                    class_colors           = class_colors_values, # used for assigning colors on pie charts to classnames, assigned alphabetically to names on 0-1 cmap scale

                    # general fig/plot aestetics
                    title                  = plot_title,
                    title_ha               = "center",
                    title_fontsize_scale   = 1.5*fontsize_scale*title_fontsize_scale, 
                    class_colors_cmap      = class_colors_cmap,
                    cmap_colors_from       = cmap_colors_from, 
                    cmap_colors_to         = cmap_colors_to,

                    # fig size & layout
                    figsze_scale           = figsize_scale,
                    figheight_scale        = figheight_scale,
                    figwidth_scale         = figwidth_scale,
                    
                    tight_lyout            = True,
                    subplots_adjust_top    = adjust_top,

                    # piecharts on each subplot
                    ax_title_fonsize_scale = 1*fontsize_scale,
                    wedges_fontsize_scale  = 1.1*fontsize_scale,
                    mid_pie_circle_color   = "lightgrey",
                    add_group_item_perc_to_numbers_in_each_pie= add_group_item_perc_to_numbers_in_each_pie,

                    # legend
                    legend_fontsize_scale  = 2*fontsize_scale,
                    legend_loc             = legend_loc,
                    legend                 = True, # because each class is annotated, 
                    legend_ncol            = legend_ncol,
                    n_subplots_in_row      = n_subplots_in_row
                )
    
    