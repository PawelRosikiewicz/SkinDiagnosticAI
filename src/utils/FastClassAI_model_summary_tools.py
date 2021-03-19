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
# from src.utils.method_comparison_tools import method_comparison_boxplot # copied here for any potential changes, 




# Function ..........................................................................
def create_class_colors_dict(*, 
    list_of_unique_names, 
    cmap_name="tab20", 
    cmap_colors_from=0, 
    cmap_colors_to=1
):
    '''Returns dictionary that maps each class name in list_of_unique_names, 
       to to a distinct RGB color
       . list_of_unique_names : list with unique, full names of clasesses, group etc..
       . cmap_name : standard mpl colormap name.
       . cmap_colors_from, cmap_colors_to, values between 0 and 1, 
         used to select range of colors in cmap, 
     '''
    
    # create cmap
    mycmap = plt.cm.get_cmap(cmap_name, len(list_of_unique_names)*10000)
    newcolors = mycmap(np.linspace(cmap_colors_from, cmap_colors_to, len(list_of_unique_names)))

    class_color_dict = dict()
    for i, un in enumerate(list_of_unique_names):
        class_color_dict[un] = newcolors[i]
    
    return class_color_dict



# Function .............................................................................
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
            
            
           
            
# Function .............................................................................             
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






# Function .............................................................................
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
    
    
    
    
    
    
    
# Function .............................................................................
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


    
    
    
    
# Function ..............................................................................
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
                
    
    

    
    
    
# Function .............................................................................  
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
  
  

  
# Function .............................................................................   
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
    
    
    
    
    
    
# Function .............................................................................
def model_summary_plot(*,
    # input data                                     
    df,
    y,
    boxname,
    boxcolor,
    scatterpoints,
    baseline,

    # fig, general settings, 
    title=None ,                               
    figsize=(30,15) ,

    # box colors
    boxcolor_dict = None,
    cmap="tab10",
    cmap_colors_from=0,
    cmap_colors_to=0.5,                               

    # axes
    xaxis_label = None,
    yaxis_label = None, # if Noene == ydata_colname
    grid_dct=dict(lw=1),                   

    # scatterpoints,
    full_model_marker ="*",
    full_model_markersize=60,
    full_model_markercolor="black",

    # legend
    add_legend=True,
    subplots_adjust_top = 0.7,
    legend_title=None,
    legend__bbox_to_anchor=(0.4, 0.9),
    legend_ncols=1,

    # baseline
    baseline_title = "baseline",
    baseline_loc =-0.09,
    use_fixed_baselines = True,
    baseline_limit_list = [0.5, 0.9, 1.5], # the last one 
    baseline_color_list = ["red", "orange", "forestgreen"],

    # fontsizes
    fontsize_scale =1,
    title_fontsize =30,
    legend_fontsize=20,
    xticks_fontsize=20,
    yticks_fontsize=20,
    axes_labels_fontsize=25,
    ):

    '''
        NGS-like boxplot for displaying accuracy, or other results obtained with large number of models
        
        # input data                                     
        df            : pd.DataFrame
        y             : str, or list with values, df colname with values to display, eg: test_accuracy ...
        boxname       : str, or list with values, df colname with values that will be displayed as names of each box, if None, 
                        (these, do not have to be unique, becaue box colors are also informative, 
                        and you may use shorter names to make the plot nicer, )
        boxcolor      : str, or list with values, if None, all boxes will hae the same colors, and there is no legend displayed, 
        scatterpoints : list, with True/False values, data points in each group used as scatter points, 
                        not the part of boxplot, if None, noe will be made, 
        baseline      : str, or list with values, df colname with values for baseline thta will be displayed on a bacground, 
        
        # horizontal patches
        use_fixed_baselines  : bool , if True, three phorizontal patches of the same height will be added to plot, 
        baseline_limit_list  : list with 3 floats, eg: [0.5, 0.9, 1.5], each float is the upper limit of the horizontal patch, 
                               starting from the plot bottom
        

    '''

    # setup
    assert type(df)==pd.DataFrame, "error: df is not pandas DataFrame"


    # . set plot x/y labels,                  
    if xaxis_label is None:
        if isinstance(boxname, str):
             xaxis_label=boxname
        else:
             xaxis_label="method"              

    if yaxis_label is None:
        if isinstance(y, str):
             yaxis_label=y
        else:
             yaxis_label="y"                                               

    # . fontsizes
    title_fontsize       = title_fontsize*fontsize_scale
    legend_fontsize      = legend_fontsize*fontsize_scale
    xticks_fontsize      = xticks_fontsize*fontsize_scale
    yticks_fontsize      = yticks_fontsize*fontsize_scale
    axes_labels_fontsize = axes_labels_fontsize*fontsize_scale
                
    # data preparation 

    # . extract columns, as list
    if isinstance(y , str):
        y = df.loc[:, y].values.tolist()
    else:
        pass

    if isinstance(boxname , str):
        boxname = df.loc[:, boxname].values.tolist()
    else:
        pass

    #. optional values, 
    if boxcolor is not None:
        if isinstance(boxcolor , str):
            boxcolor = df.loc[:, boxcolor].values.tolist()
        else:
            pass
    else:
        boxcolor = ["method"]*len(y)

    if baseline is not None:
        if isinstance(baseline , str):
            baseline = df.loc[:, baseline].values.tolist()
        else:
            pass
    else:
        baseline = [0.5]*len(y)

    if scatterpoints is not None:
        if isinstance(scatterpoints , str):
            scatterpoints = df.loc[:, scatterpoints].values.tolist()
        else:
            pass
    else:
        scatterpoints = [False]*len(y) # ie, No data wil be plotted as scatter point, 

    # . create unique boxnames qwith colors and method names, 
    if boxcolor is not None:
        boxname_full  = [f"{x}__{y}" for (x,y) in zip (boxname, boxcolor)] # used to search values, 
    else:
        boxname_full = boxname


    # assign colors to each boxcolor name

    # . define colors for each class in boccolor
    if boxcolor_dict is None:
        boxcolor_dict = create_class_colors_dict(
                list_of_unique_names = pd.Series(boxcolor).unique().tolist(), 
                cmap_name            = cmap, 
                cmap_colors_from     = cmap_colors_from, 
                cmap_colors_to       = cmap_colors_to
            )    
    else:
        pass

    # . map colors onto boxcolor, that are names
    boxcolor_value = pd.Series(boxcolor).map(boxcolor_dict)

    # build pandas df wiht all data
    boxplotdf = pd.DataFrame({
        "y": y,                               # value on y-axis
        "boxname_full": boxname_full,         # used to separate each box (combines x-axis anme and color)
        "boxcolor_value": boxcolor_value,     # color for bocplot, 
        "boxname":boxname,                    # displayed on x axis, 
        "boxcolor":boxcolor,                  # displayed on legend, 
        "baseline": baseline,                 # displayed as bacground value, 
        "scatterpoints": scatterpoints,       # it True, the point is plotted as scatterplot, 
    })

    # data preparation - part 2 - prepare array and ncols for plot

    # . lists with data for boxes, 
    'one item for one box in each'
    x_axis_name    = [] # xtick labels
    x_axis_color   = [] # xtick label color
    bx_x           = []
    bx_y           = [] 
    bx_color       = [] # box color, (only for boxes)
    sc_y           = []
    sc_x           = []
    baseline_x     = []
    baseline_y     = []
    median_y       = []

    # . fill in values, in proper order with positons on x axis, 
    for i, one_boxname_full in enumerate(pd.Series(boxname_full).unique().tolist()):

        # find data for boxes
        boxplotdf_bx_subset = boxplotdf.loc[(boxplotdf.boxname_full==one_boxname_full) & (boxplotdf.scatterpoints==False), :]
        if boxplotdf_bx_subset.shape[0]>0:
            bx_x.append(i)
            bx_y.append(boxplotdf_bx_subset.loc[:,"y"].values.tolist())
            bx_color.append(boxplotdf_bx_subset.boxcolor_value.iloc[0])
        else:
            pass

        # find data for scatter points, 
        boxplotdf_sc_subset = boxplotdf.loc[(boxplotdf.boxname_full==one_boxname_full) & (boxplotdf.scatterpoints==True), :]
        sc_values = boxplotdf_sc_subset.loc[:,"y"].values.tolist()
        if len(sc_values)>0:
            sc_x.extend([i]*len(sc_values))
            sc_y.extend(sc_values)  
        else:
            pass

        # axis_name, baseline, 
        boxplotdf_group_subset = boxplotdf.loc[boxplotdf.boxname_full==one_boxname_full, :]
        baseline_x.append(i)
        baseline_y.append(boxplotdf_group_subset.baseline.max())  
        median_y.append(boxplotdf_group_subset.y.median())
        x_axis_name.append(boxplotdf_group_subset.boxname.iloc[0])
        x_axis_color.append(boxplotdf_group_subset.boxcolor_value.iloc[0])


    # order items on x axis, 

    # . dict with key == old postion, value == new postion
    '''
        I am using dict, because each item may have different number of 
        elements, and they are not in order, (ie one category may be nmissing and present in sc or bx)
        that is completely normal !
    '''
    x_order    = dict(zip(pd.Series(median_y).sort_values().index.values.tolist(), list(range(len(median_y)))))
    bx_x       = pd.Series(bx_x).map(x_order).values.tolist()
    sc_x       = pd.Series(sc_x).map(x_order).values.tolist()
    baseline_x = pd.Series(baseline_x).map(x_order).values.tolist()

    # . created ordered_xticks_labels
    tempdf = pd.concat([pd.Series(median_y), pd.Series(x_axis_color), pd.Series(x_axis_name),  pd.Series(bx_color), pd.Series(bx_y)], axis=1)
    tempdf.columns=["median", "x_axis_color","x_axis_name", "bx_color", "by"]
    tempdf = tempdf.sort_values("median")
    tempdf.reset_index(drop=True, inplace=True)
    ordered_xticks_labels = tempdf.x_axis_name.values.tolist()
    ordered_xticks_colors = tempdf.x_axis_color.values.tolist()
    ordered_bx_color = tempdf.bx_color.dropna().values.tolist()
    ordered_b = tempdf.by.dropna().values.tolist()

    # . add small gausion noise to sc_x positions, 
    sc_x = (np.random.normal(loc=0, scale=0.05, size=len(sc_x))+np.array(sc_x)).tolist()

    
    
    # figure
    
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        pass
        
    # boxplots
    bx = ax.boxplot(
        bx_y,
        positions=bx_x,
        showfliers=True,                  # remove outliers, because we are interested in a general trend,
        vert=True,                        # boxes are vertical
        patch_artist=True,
        widths=0.3
    )
    
    # . add colors to each box individually,
    for i, j in zip(range(len(bx['boxes'])),range(0, len(bx['caps']), 2)) :
        median_color  ="black"
        box_color     = bx_color[i]
        
        # set properties of items with the same number as boxes,
        plt.setp(bx['boxes'][i], color=box_color, facecolor=median_color, linewidth=2, alpha=0.8)
        plt.setp(bx["medians"][i], color=median_color, linewidth=2)
        plt.setp(bx["fliers"][i], markeredgecolor="black", marker=".") # outliers

        # set properties of items with the 2x number of features as boxes,
        plt.setp(bx['caps'][j], color=median_color)
        plt.setp(bx['caps'][j+1], color=median_color)
        plt.setp(bx['whiskers'][j], color=median_color)
        plt.setp(bx['whiskers'][j+1], color=median_color)
        
    # points, 
    if pd.Series(scatterpoints).sum()>0:
        ax.scatter(
            x=sc_x, 
            y=sc_y,
            color=full_model_markercolor, 
            s=full_model_markersize,  
            marker=full_model_marker, 
            zorder=100
        )
    else:
        pass    

    # general aestetics, 
    ax.set_xlim(-0.5, len(x_axis_name)-0.5)
    ax.grid(ls=":",  color="grey", **grid_dct)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(list(range(len(ordered_xticks_labels))))
    ax.set_xticklabels(ordered_xticks_labels, rotation=45, fontsize=xticks_fontsize, ha="right")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=yticks_fontsize)
    ax.set_ylabel(yaxis_label, fontsize=axes_labels_fontsize)
    ax.set_xlabel(xaxis_label, fontsize=axes_labels_fontsize)
    ax.set_ylim(0,1.02)

    # . add the same color to xtick labels as it is given to the gorup 
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(ordered_xticks_colors[i])

    # legend,    
    if add_legend==True:
        
        # . create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
        sorted_keys = pd.Series(list(boxcolor_dict.keys())).sort_values().values.tolist()
        patch_list_for_legend =[]
        for i, m_name in enumerate(sorted_keys):
            label_text = f"{m_name}"
            patch_list_for_legend.append(mpl.patches.Patch(color=boxcolor_dict[m_name], label=label_text))

        # . add patches to plot,
        leg = fig.legend(
                handles       = patch_list_for_legend, 
                frameon       = False, 
                title         = legend_title,
                scatterpoints = 1, 
                ncol          = legend_ncols, 
                bbox_to_anchor= legend__bbox_to_anchor, 
                fontsize      = legend_fontsize
                )
        plt.setp(leg.get_title(),fontsize=legend_fontsize)#'xx-small')

        # . create space for the legend
        fig.subplots_adjust(top=subplots_adjust_top)    
    
    else:
        pass

    # color patches behing boxplots, 
    patch_width = 1   # ie. 1 = grey patch for 1 and 1 break
    patch_color = "grey"
    pathces_starting_x = list(range(0, len(x_axis_name), patch_width*2))
    for i, sx in enumerate(pathces_starting_x):
        rect = plt.Rectangle(
                (sx+0.5, 0), 
                patch_width, 
                1000, 
                color=patch_color, 
                alpha=0.4, 
                edgecolor=None
                )
        ax.add_patch(rect)        

    # horizontal color patches, 
    if use_fixed_baselines==False:
        params = dict(width=1, alpha=0.15, edgecolor=None)

        # create color pathes corresponding to baseline in each method
        for i, (bx, by) in enumerate(zip(baseline_x, baseline_y)):
            rect = plt.Rectangle((bx-0.5,0), height=by, color=baseline_color_list[0], **params)
            ax.add_patch(rect)              
            rect = plt.Rectangle((bx-0.5,by), height=baseline_limit_list[1]-by, color=baseline_color_list[1], **params)
            ax.add_patch(rect)             
            rect = plt.Rectangle((bx-0.5,baseline_limit_list[1]), height=1000, color=baseline_color_list[2], **params)
            ax.add_patch(rect) 

    else:       
        params = dict(width=len(baseline_x)*100, alpha=0.2, edgecolor=None)

        rect = plt.Rectangle((-0.5,0), height=baseline_limit_list[0], color=baseline_color_list[0], **params)
        ax.add_patch(rect)          
        rect = plt.Rectangle((-0.5,baseline_limit_list[0]), height=baseline_limit_list[1]-baseline_limit_list[0], color=baseline_color_list[1], **params)
        ax.add_patch(rect)             
        rect = plt.Rectangle((-0.5,baseline_limit_list[1]), height=1000, color=baseline_color_list[2], **params)
        ax.add_patch(rect)               

        # add line with baseline
        if baseline_title is not None:
            ax.axhline(baseline_limit_list[0], lw=2, ls="--", color="dimgrey")
            ax.text(len(baseline_x)-2, baseline_limit_list[0]+baseline_loc, baseline_title, ha="right", color="dimgrey", fontsize=yticks_fontsize)        
        else:
            pass
     
    return fig

