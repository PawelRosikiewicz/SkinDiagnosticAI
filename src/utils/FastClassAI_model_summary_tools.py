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


from src.utils.model_summary_plots import  visual_model_summary
from src.utils.method_comparison_tools import method_comparison_boxplot




# Function ...................................................................................................... 
def load_summary_files(*,
    dataset_name,
    dataset_variants,
    module_names,
    ai_methods,
    keywords,
    path_results,
    verbose=False                  
):

    # assure that you have proper datastructures 
    if isinstance(dataset_variants, str):
        dataset_variants = [dataset_variants]
    else:
        pass
    if isinstance(module_names, str):
        module_names = [module_names]
    else:
        pass
    if isinstance(ai_methods, str):
        ai_methods = [ai_methods]
    else:
        pass
    if isinstance(keywords, str):
        keywords = [keywords]
    else:
        pass


    # collect names of files that will be loaded
    file_counter=0
    for ai_method in ai_methods:
        for dataset_variant in dataset_variants:
            for module_name in module_names:

                if verbose==True:
                    print("Loading files for: ", ai_method, dataset_variant, module_name, "Found: ", end="")
                else:
                    pass
                
                # path
                rpath = os.path.join(path_results, f"{ai_method}__{dataset_name}__{dataset_variant}")
                os.chdir(rpath)

                # find all files in rpath
                files = []
                for file in glob.glob("*"):
                    files.append(file)

                # select all with keywords, 
                files_s = pd.Series(files)  
                for k in keywords:
                    files_s = files_s.loc[files_s.str.contains(k)]
                files_l = files_s.values.tolist()    

                # info part 2,
                if verbose==True:
                    print(len(files_s)>0, "files")
                else:
                    pass
                
                # load files
                if len(files_s)>0:
                    for file_name in files_l :
                        loaded_df = pd.read_csv(file_name)
                        loaded_df["file_name"]=[file_name]*loaded_df.shape[0]
                        loaded_df["path"]=[rpath]*loaded_df.shape[0]

                        if file_counter==0:
                            summary_df = loaded_df
                            file_counter += 1
                        else:
                            summary_df = pd.concat([summary_df, loaded_df], axis=0)
                            summary_df.reset_index(inplace=True, drop=True)  
                else:
                    pass

    # info part 2,
    if verbose==True:
        print("----> Final Table has results for ", summary_df.shape[0], " models")
    else:
        pass      
    
    return summary_df
            
            
           
            
# Function, .................................................................... ..................................             
def create_new_df_feature(*, df, new_feature_name, old_features_list, return_full_df=True, verbose=False):
    '''
        create new feature by concatanating corresponsing cells 
        in pd dataframe form any number of other selected features
        
        old_features_list: str, or list, with name/s of feature to be concatenated
        return_full_df   : bool, if True entire df is retuned
                           if False, return pd.series only with thenew feature
        
    '''
    if isinstance(old_features_list, str):
        old_features_list = [old_features_list]
    else:
        pass
    
    # check if all feaqtures are available
    stop_the_function = False
    for i, feature in enumerate(old_features_list):
        try:
            df.loc[:, feature]
        except:
            stop_the_function = True
            if verbose==True:
                print(f"ERROR: {feature} -- was not found in dataframe")
            else:
                pass
            
    # concatanate values in each corresponding cell
    if stop_the_function==True:
        return None
        
    else:
        for i, feature in enumerate(old_features_list):
            if i==0:
                new_feature = df.loc[:, feature].values.tolist()
            else:
                another_new_feature = df.loc[:, feature].values.tolist()
                new_feature = [f"{x}__{y}" for (x,y) in zip(new_feature, another_new_feature)]
        
        
        if return_full_df==True:
            df[new_feature_name] = new_feature
            return df
        
        else:
            return pd.Series(new_feature)






# Function ......................................................................................................
def simple_visual_model_summary(*, 
            model_idx_in_sorted_summary_df=0,
            subset_collection_name,
            batch_names_list,
            summary_df,
            class_labels_configs,
            path_data,
            path_results,   
            N_displayed_images ="all",
            max_img_per_col = 15,
            fontsize_scale= 1 
): 
    '''
        Temporary function used only with FastClassAI pipeline, that will load raw images 
        from all batches in a given subset batch collection, eg batch 1 and 2 for test data, 
        then it will plot 3 figures
        - 1st figure - images grouped by class assigned with the model and with color boxes showing true class
        - 2nd/3rd figure - pie charts showing sencsitivity and specificity in pie charts
        
        PS: I am working on better version
    '''
    
    # ....
    idx_in_df = model_idx_in_sorted_summary_df

    # *** find names and models to load
    # sort summary df
    sorted_summary_df = summary_df.sort_values('model_acc_valid', ascending=False)
    sorted_summary_df.reset_index(inplace=True, drop=True)

    # get all variables, 
    method                        = sorted_summary_df.method.iloc[idx_in_df]
    dataset_name                  = sorted_summary_df.dataset_name.iloc[idx_in_df]
    dataset_variant               = sorted_summary_df.dataset_variant.iloc[idx_in_df]
    model_ID                      = sorted_summary_df.model_ID.iloc[idx_in_df] # its an ID number given to the model in that dictionary, 


    # *** paths
    path_to_raw_images_sorted_into_batches = os.path.join(path_data, f'{dataset_name}__{dataset_variant}')
    path_to_batch_labels  = os.path.join(path_data, f'{dataset_name}__{dataset_variant}__extracted_features')
    path_to_model_predictions = os.path.join(path_results, f'{method}__{dataset_name}__{dataset_variant}')


    # *** load data
    # load model predictions, 
    os.chdir(path_to_model_predictions)
    model_predictions_file_name   = re.sub("summary_table.csv", "model_predictions_dict.p", sorted_summary_df.file_name.iloc[idx_in_df])
    with open(model_predictions_file_name , 'rb') as file: 
        model_predictions_dict  = pickle.load(file)  

    # get class_label colors, 
    class_labels_colors_toUse     = class_labels_configs[dataset_variant]['class_labels_colors']


    # caulate accuracy results
    acc_results = f'acc={np.round(sorted_summary_df.loc[:, f"model_acc_{subset_collection_name}"].iloc[idx_in_df],2)}'

    # display examples from best performing model,
    visual_model_summary( 
            model_predictions_dict  = model_predictions_dict, 
            model_ID                = model_ID,    # applicable only with a given model_predictions_dict

            # what predicitons to display, 
            n                       = N_displayed_images, # use "all" to display all 
            examples_to_plot        = "all", # correct and incorrect on the same plot, 
            class_colors            = class_labels_colors_toUse,

            # input data, 
            dataset_name            = dataset_name,                              
            subset_name             = [subset_collection_name],           # name used in xy_names eg: train, valid, test test_2
            img_batch_subset_names  = batch_names_list,                   # list, batch names that were placed in that collection, 
            path_to_raw_img_batch   = path_to_raw_images_sorted_into_batches, 

            # ... settings for main plot, 
            title_prefix                 = f"{subset_collection_name}, {acc_results}",
            make_plot_with_img_examples  = True, # use False, to have only pie charts with classyfication summary                                         
            add_proba_values_to_img_name = True,
            max_img_per_col              = max_img_per_col,

            # ... settings for annot. pie charts, 
            first_pie_title              =f"Image Classyfication Results - True Class in pie chart ring\n{subset_collection_name} data",
            second_pie_title             =f"Class Detection Results - True Class in pie chart center \n{subset_collection_name} data",
            pie_data_for_all_images_in_img_batch=True,
            pie_charts_in_ne_row         = 7,

            # ... pie chart aestetics added later to tune pie charts
            PIE_legend_loc = "upper right",
            PIE_ax_title_fonsize_scale=0.6*fontsize_scale,
            PIE_legend_fontsize_scale=1.4*fontsize_scale,
            PIE_wedges_fontsize_scale=1*fontsize_scale,
            PIE_legend_ncol=4,
            PIE_tight_lyout=False,
            PIE_title_ha="right",
            PIE_figsze_scale=1.5,
            PIE_subplots_adjust_top=0.75,
            PIE_ax_title_fontcolor="black"
        )
    
    
    
    
    
    
    
# Function ...................................................................................................... 
def prepare_summary_df(*,
        dataset_name,
        dataset_variants,
        module_names,
        ai_methods,
        keywords,
        path_results,
        verbose=False                            
):
    '''
        helper function that loads results from model evaluation, 
        for all combinaiton of dataset_name, dataset_variants, module_names, ai_methods
        and keywords, that allow to find one or more csv file names (order of keywords is not important)
        it will provide only files wiht exact match for all keywords, if nothing is returned, set verbose==True, 
        
        ai_method, dataset_name, and dataset_variants, are used to build folder names in path_results
        whereas keywords and module names are used to find files 
        
        caution, the function load_summary_files requires module names for iteration, but these are not used to find files, 
        it was an error that will be removed, if required results for speciffic module, place it name in keywords,
        and only files created for that module will be loaded
    '''
    
    
    summary_df = load_summary_files(
        dataset_name         = dataset_name,
        dataset_variants     = dataset_variants,
        module_names         = module_names,
        ai_methods           = ai_methods,
        keywords             = keywords,
        path_results         = path_results,
        verbose              = verbose               
    )

    summary_df = create_new_df_feature(
        df = summary_df, 
        new_feature_name = "full_method_name", 
        old_features_list = ["method", "method_variant"],
    )
    summary_df = create_new_df_feature(
        df = summary_df, 
        new_feature_name = "full_dataset_variant", 
        old_features_list = ["dataset_variant", 'module'],
        verbose=True
    )
    summary_df = create_new_df_feature(
        df = summary_df, 
        new_feature_name = "full_results_group_name", 
        old_features_list = ["method", "method_variant", "dataset_variant", 'module'],
    )
    
    return summary_df


    
    
    
    

    
# Function ......................................................................................................    
def create_boxplot_with_color_classes(*,
    summary_df, 
    figsize              = (10,6),
    col_with_results     ="model_acc_valid",       # df colname with values to display, eg: test_accuracy ...
    col_with_group_names ="full_method_name",           # df colname with values that will be displayed as names of each box (these do not have to be unique)
    col_with_group_ID    ="full_results_group_name",      # df colname with values that will be grouped for separate boxes (must be unieque)
    col_with_group_colors="full_dataset_variant",       # df colname with values that will have different colors (colors can not be mixed within diffeent group_ID)
    baseline = 0.5,
    fontsize_scale=1, 
    subplots_adjust_top = 0.6,
    baseline_title ="baseline",
    legend_ncols=1,
    legend__bbox_to_anchor=(0.5, 1.1),
                                     
):
    '''
        Funtion returns boxplot showing accuracy or other metric displayed for any number 
        of groups of results (eg methods), divided into larger groups shown as different colors of boyes, 
        with color legen above the plot
    
        summary_df           : summary dataframe created with prepare_summary_df function, 
        col_with_results     : str, df colname with values to display, eg: test_accuracy ...
        col_with_group_names : str, df colname with values that will be displayed as 
                               names of each box (these do not have to be unique)
        col_with_group_ID    : str, df colname with values that will be grouped for 
                               separate boxes (must be unieque)
        col_with_group_colors: str, df colname with values that will have different colors 
                               (colors can not be mixed within diffeent group_ID)
    
    '''


    # boxplot
    fig = method_comparison_boxplot(
            title=f"Accuracy of models created with each method\n\n",  
            data = summary_df,         # pd.DataFrame with the results,   
            figsize=figsize,
            # ...
            col_with_results     =col_with_results,       # df colname with values to display, eg: test_accuracy ...
            col_with_group_names =col_with_group_names ,           # df colname with values that will be displayed as names of each box (these do not have to be unique)
            col_with_group_ID    =col_with_group_ID,      # df colname with values that will be grouped for separate boxes (must be unieque)
            col_with_group_colors=col_with_group_colors,       # df colname with values that will have different colors (colors can not be mixed within diffeent group_ID)
            # ... colors
            cmap="tab10",
            cmap_colors_from=0, 
            cmap_colors_to=0.5,                               
            # .. legend
            legend__bbox_to_anchor=(0.5, 1.1), 
            subplots_adjust_top = subplots_adjust_top,
            legend_ncols=legend_ncols,
            # .. baseline
            baseline_title =baseline_title,
            baseline_loc =-0.09,
            baseline = baseline,          
            top_results = 0.9,      # green zone on a plot, 
            # ... fontsize
            title_fontsize=20*fontsize_scale,
            legend_fontsize=10*fontsize_scale,
            xticks_fontsize=10*fontsize_scale,
            yticks_fontsize=15*fontsize_scale,
            axes_labels_fontsize=15*fontsize_scale,
            # ... axies labels
            xaxis_label = "Method",
            yaxis_label = "Accuracy\n",
            paint_xticks=True
        )

    return fig
  
  
  
 



# Function ......................................................................................................    
def preapre_table_with_n_best_results_in_each_group(*,
    summary_df,              
    n_top_methods = 1,
    sort_by = "model_acc_valid",
    feature_used_to_group_models = "full_results_group_name",
    display_table=False 
):
    '''
        Function that takes summary df, selectes max n requested best perfoming models, 
        and return them all in sorted summary df table format, 
        if display_table==True, displays selected columns from that table to show all examples, 
    '''
    

    # unique model group names 
    method_full_name_list   = summary_df.loc[:, feature_used_to_group_models].unique().tolist()

    # collect top methods, 
    for i, method_full_name in enumerate(method_full_name_list):

        # . subset summary_df
        summary_df_subset = summary_df.loc[summary_df.loc[:, feature_used_to_group_models]==method_full_name, :]
        summary_df_subset = summary_df_subset.sort_values(sort_by, ascending=False)

        # . place in 
        if i==0:
            best_methods_summary_df = summary_df_subset.iloc[0:n_top_methods,:]
        else:
            best_methods_summary_df = pd.concat([best_methods_summary_df, summary_df_subset.iloc[0:n_top_methods,:]])
        best_methods_summary_df.reset_index(drop=True, inplace=True) 

    # display examples:
    # show best model examples
    features_to_display = ["dataset_variant", "module","method", "method_variant", 
                               "model_acc_train", "model_acc_valid", "model_acc_test", 
                           "pca_components_used", "run_name"]
    sorted_best_methods_summary_df = best_methods_summary_df.sort_values("model_acc_valid", ascending=False)
    sorted_best_methods_summary_df.reset_index(drop=True, inplace=True)
    
    if display_table==True:
        features_to_display = ["dataset_variant", "module","method", "method_variant", 
                                   "model_acc_train", "model_acc_valid", "model_acc_test", "baseline_acc_test",
                               "pca_components_used", "run_name"]
        display(sorted_best_methods_summary_df.loc[:, features_to_display])
    else:
        pass
    
    return sorted_best_methods_summary_df
    