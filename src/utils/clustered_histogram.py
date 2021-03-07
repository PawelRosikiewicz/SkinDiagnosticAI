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



# ************************************************************************* #
#     Author:   Pawel Rosikiewicz                                           #       
#     Copyrith: IT IS NOT ALLOWED TO COPY OR TO DISTRIBUTE                  #
#               these file without written                                  #
#               persmission of the Author                                   #
#     Contact:  prosikiewicz@gmail.com                                      #
#                                                                           #
# ************************************************************************* #


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





# Function ...........................................................................
def find_n_examples_in_each_class(*, pdseries, n=4):
    """retruns df with n indexes of evently spaced examples of each class
       where the 1st column is a class name, and each following column is 
       one example from each class, 
       if NA, then there is no more examples, or it was filled with repeats
       
        pdseries :  pd series with class names, 
                    eg: ["water", "water", "soil", "soil"...] 
                    it will return n, exacples of each class ("wter and soil", in the above)
                    that are evenly spaced among instances of each class,
                    but ordered as in the original pd.series,
                    the reruned dataframe will contain index numbers, for these examples, 
                    
        Caution !  if from some reason, classses were reordered, eg by clusting, and you wish to get examples, 
                   the original, index shoudl be set in pdseries index, 
    """
    
    class_names = list(pdseries.unique())
    class_examples = pd.DataFrame(np.zeros([len(class_names), n+1])) # +1 col, for class_names, 
    
    #.. get evenly spaced examples of each class as ordered on the dendrogram, 
    for i, oneclass in enumerate(class_names):
        
        # find how many images you have in oneclass, and select evenly spaces, if possible, 
        idx_list = list(pdseries[pdseries==oneclass].index)
        idx_list = list(range(len(idx_list)))
        if int(len(idx_list))<n: 
            idx_list.extend([np.nan]*np.abs(n-len(idx_list)))
        elif int(len(idx_list))==n: 
            idx_list = idx_list
        elif int(len(idx_list))>n: 
            idx_list = [int(x) for x in list(np.floor(np.linspace(0, len(idx_list)-1, n)))]

        # get img indexes in the batch 
        img_idx_examples = list()
        for idx in idx_list:
            if np.isnan(idx): img_idx_examples.append(np.nan) # add nan if there is no nore images to add.
            else: img_idx_examples.append(list(pdseries[pdseries==oneclass].index)[idx]) # add index in img_batch of img example,

        # place them inside the class_examples df, 
        class_examples.iloc[i, 0] = oneclass
        class_examples.iloc[i, 1:n+1] = img_idx_examples

    #.. set class names as index
    class_examples = class_examples.set_index(0)

    # re-order rows, in class_examples so that, the 1st class from pdseries, is in 1st row, 2nd in 2nd rows etc...
    pdseries_forordering = pdseries.copy() # just to work on copy, 
    pdseries_forordering.reset_index(drop=True, inplace=True) # in case, pd.series was 
    
    #.. add class name to order list, but only if it is new for that list, 
    order_list = [pdseries_forordering[0]]
    for i in  pdseries_forordering:
        if ((pd.Series(order_list)==i).sum()>0): pass
        if ((pd.Series(order_list)==i).sum()==0): order_list.append(i)
          
    #.. reorder rows in class_examples, 
    class_examples = class_examples.loc[order_list,:]
    
    # .. because I had some problems, i extract and then remove class_names from my df, 
    ordered_class_names = list(class_examples.index)
    class_examples = class_examples.reset_index(drop=True)
    
    return (class_examples, ordered_class_names)





  
  
  
  

# Function, .............................................................................
def clustered_histogram_with_image_examples(*,
    encoded_img_batch, batch_labels, raw_img_batch, class_colors, 
    plot_title="", method_name = "", row_linkage=None, legend_ncol=4,
    select_features=True, use_categorical_features=False, add_cluster_description=False,
    show_first=False, verbose=False,
    ):

    """
      Plots clustered dendrogram for images stored in one batch, with features extracted with tfhub module,
      and up to four examples of images of up to six classes that appearted at first on image dendrogram, 
      
      # Input
        -----------------------------------------------------------------------------------
      . encoded_img_batch  : numpy array, [<img number>, <feature number>] 
      . batch_labels       : dataframe, created with encode_images_with_tfhubmodule, 
                             from feature_extraction_tools
      . raw_img_batch      : numpy array, [?, img_width, img_height, 3], 
                             works only with scalled (1/255) rgb images,
      . class_colors       : dictionary, 
                             {str <"class_name">: str <"color">}
                             CAUTION: colors and class names must be unique !
      . plot_title         : str, on top of the plot,  
      . method_name        : str, added at the end of a-axis description
      . hist_cmap          : str, eg: "plasma"
      . add_cluster_description : bool, if true, a more descirptive name will be added to each cluster on top of heatmap, howevr it may be long and look ugly, 
      . legend_ncol        : int >1, ncol, paramter in legend function from matplotlib.pyplot
      
        ..
      . use_categorical_features :bool, if, True extracted feauture values will be encoded 
                             as 1,2,3, for (0,1], (1, 2], and >2, values respectively,  
      . select_features    : bool, if True, the heatmap, and dendrogram will be constructed 
                             only using features that were different in at least one image
      . show_first         : bool, if True, the plot will display image examples from 
                             up to 6 claseses that appeared at first on dendrogram, 
                             if False, the examples will be selected from up to six
                             classes with the largest number of images,
      # Returns,
        -----------------------------------------------------------------------------------
      . Figure,            : shows the figure by default, with:
      . dictionary           1. "image_order_on_dendrogram" - pd.dataframe, 
                             2. "plotted_image_examples - pd.dataframe  
      # Comments:
        -----------------------------------------------------------------------------------
        Potential issues, or usefull topics for future developments, 
        -   rotating yticks on clustergrid
            https://stackoverflow.com/questions/34572177/labels-for-clustermap-in-seaborn
        -   setting y/x-axis labels
            https://stackoverflow.com/questions/41511334/adding-text-to-each-subplot-in-seaborn
        -   how to set heatmap and cluster dendrogram dimensions on the plot
            https://stackoverflow.com/questions/33137909/changing-the-size-of-the-heatmap-specifically-in-a-seaborn-clustermap

       Potential problem, 
        -   accently, I used class color as class name in some part of the code (drwaing image examples)
            this may be a problem if two or more classes have the same color, or you dont use "str" do decribe the color, 
            remove that with the next update !
    
    """

    # default params:
    figsize = (20,10) # i tried different sizes, this one looks best !
                      # chn age it on your own responsability :P
    max_nr_of_classses_to_display_examples = 6
    max_nr_of_examples_from_each_class_to_display = 4
    cluster_dividing_line_color = 'black'
    cluster_dividing_line_style = '--'
    cluster_dividing_line_width = 2
    cluster_name_fontsize = 15
    cluster_name_color = cluster_dividing_line_color
    heatmap_cmap = "Wistia"

        
    # work on copies, 
    encoded_img_batch = encoded_img_batch.copy()
    batch_labels = batch_labels.copy()
    raw_img_batch = raw_img_batch.copy()        

    

    # .............................................................................
    # prepare the data for plot, 

    
    # optional features trnaformation into categorical data,  
    
    # simplify values, to allow faster plotting and clusterring,  
    simplefeatures = encoded_img_batch.copy()
    simplefeatures[encoded_img_batch<=1]=1
    simplefeatures[np.logical_and(encoded_img_batch>1, encoded_img_batch<=2)]=2
    simplefeatures[encoded_img_batch>2]=3

    # find variable features 
    feature_sums = simplefeatures.sum(axis=0)
    selector = np.logical_or(
                    feature_sums==1*simplefeatures.shape[0], 
                    feature_sums==2*simplefeatures.shape[0],
                    feature_sums==3*simplefeatures.shape[0],
                )==False

    # use raw or simplified/sharpenned features to create heatmap
    if use_categorical_features==False:
        features_to_plot = encoded_img_batch
    else:
        features_to_plot = simplefeatures

    # remove all features that are uniform across all encoded images, 
    if select_features==True:
        features_to_plot = features_to_plot[:, selector] 
    else:
        pass    

    # calculate stats to display in plot title, 
    featurenumber = selector.shape[0]
    variablefeatures = selector.sum()
    usedfeatures = features_to_plot.shape[1] 

    
    # hierarchical clustering, 

    # calculate chierarchical clustering on x/y axses or use the one provided with arguments,  
    correlations_array = np.asarray(features_to_plot)
    col_linkage = hierarchy.linkage( distance.pdist(correlations_array.T), method='ward')
    if row_linkage is None:
        row_linkage = hierarchy.linkage( distance.pdist(correlations_array), method='ward')
        row_order = leaves_list(row_linkage) 
    else:
        # row_linkage provided with list of arguments, 
        row_order = leaves_list(row_linkage) 
    

    # .............................................................................
    # select class colors to display and mapt them to the class instances in batch_labels

    # set colors for classes displayed on yaxis, after class_colors,
    color_mix = dict(zip(list(class_colors.keys()), list(class_colors.values())))
    row_colors = pd.Series(batch_labels.classname.values.tolist()).map(color_mix)
    ordered_row_colors = row_colors[row_order]    
    
    
    
    
    
    # .............................................................................
    # select image examples to plot,
    
    # print(max_nr_of_examples_from_each_class_to_display)
    # print(batch_labels.classname[row_order])

    # Identify up to four examples from each class, that are evenly spaces across that clus in order created by hierarchical clustering, 
    img_examples_to_plot, ordered_class_names_examples = find_n_examples_in_each_class(
        pdseries=batch_labels.classname[row_order], 
        n=max_nr_of_examples_from_each_class_to_display)
    img_examples_to_plot.index=ordered_class_names_examples # classes will appear in that order
    # ..
    ordered_colors_for_class_names_examples = list()
    for cn in ordered_class_names_examples:
        ordered_colors_for_class_names_examples.append(class_colors[cn])
    
    # prepare small df, to help locating labels later on, (df with one column with class names, and 2 columns with original and dendrogram indexing)
    ordered_class_names_with_dedrogram_numbering = batch_labels.classname[row_order].reset_index(drop=False)
    ordered_class_names_with_dedrogram_numbering.reset_index(drop=False, inplace=True)
    ordered_class_names_with_dedrogram_numbering.columns = ["idx_in_dedrogram", "idx_in_img_batch", "classname"]
    ordered_class_names_with_dedrogram_numbering["color_assigned"] = ordered_row_colors.reset_index(drop=True)
        
    # because of space constrain, you can plot image examples from up to six classes,    
    if len(ordered_class_names_examples)<=max_nr_of_classses_to_display_examples:
        selected_classes_to_plot = list(img_examples_to_plot.index)
        
    else:
        if show_first==True:
            selected_classes_to_plot = list(img_examples_to_plot.index)[0:max_nr_of_classses_to_display_examples]

        if show_first!=True:            
            # I am selecting classes with the largest nunmber of images, in that case several items must be modified, 
            counts_per_class = batch_labels.groupby("classname").count().iloc[:,0].sort_values(ascending=False)
            classes_with_the_largest_number_of_images = list(counts_per_class.index)[0:max_nr_of_classses_to_display_examples]
            
            #.. select these classes, 
            # I am selecting classes with the largest nunmber of images, in that case several items must be modified, 
            selected_classes_to_plot = list()
            for cn in ordered_class_names_examples:
                if (np.array(classes_with_the_largest_number_of_images)==cn).sum()==1:
                    selected_classes_to_plot.append(cn)
                else:
                    pass       
            selected_img_examples_to_plot =  img_examples_to_plot.loc[selected_classes_to_plot,:]   
    # ..
    selected_img_examples_to_plot = img_examples_to_plot.loc[selected_classes_to_plot,:]
    #..
    if verbose==True:
        print(f"Examples from the following classes will be plotted: {selected_classes_to_plot}")
   
        
    # .............................................................................
    # Main Figure: seaborn, clustered heatmap

    # Create clustered heatmap,
    sns.set()
    g = sns.clustermap( 
        pd.DataFrame(features_to_plot), 
        row_colors=row_colors, 
        cmap=heatmap_cmap,
        row_linkage=row_linkage, 
        col_linkage=col_linkage, 
        method="average",
        xticklabels=False, 
        figsize=figsize,
        yticklabels=True,
        alpha=1
        )

    # figure title and axes decription, 
    g.fig.suptitle(f'{plot_title}', fontsize=30)  
    
    # xaxis dendrogram,
    g.fig.axes[2].set_ylabel("\nCluestered Images\n\n\n\n\n\n", fontsize=20)

    # heatmap,
    g.fig.axes[3].set_xlabel(f"Clustered features extracted from images {method_name}", fontsize=20)

    # small histogram legend,
    g.fig.axes[4].set_title("Heatmap\nFeature Values")

    # collect tick labels for later on
    img_idx_tick_labels = list()
    for i, tick_label in enumerate(g.ax_heatmap.axes.get_yticklabels()):
        # collect original labels, 
        img_idx_tick_labels.append(tick_label.get_text())
        tick_text = tick_label.get_text()
        tick_label.set_color("white") # so it disaapears and do not interfier with custom labels, 
        
        
        
        
    # .............................................................................
    # Main Legend - later on, I wish to modify that part, 

    # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
    patch_list_for_legend =[]
    count_items = 0
    for i, cl_name in enumerate(list(selected_img_examples_to_plot.index.values)):
        cl_color = class_colors[cl_name]
        if i<17: 
            class_number_in_the_batch = (batch_labels.classname==cl_name).sum()
            label_text = f"{cl_name}; {class_number_in_the_batch} ({np.round((class_number_in_the_batch/features_to_plot.shape[0]*100),0)}%)"
            patch_list_for_legend.append(mpatches.Patch(color=cl_color, label=label_text))
        if i==17: 
            patch_list_for_legend.append(mpatches.Patch(color="white", label=f"+ {selected_img_examples_to_plot.shape[0]} classes in dataset... "))
        if i>17: 
            break # ie, when the legend is onger then 3 lines
               
    # add patches to plot,
    l = g.fig.legend(handles=patch_list_for_legend, 
        loc="center", frameon=False, 
        scatterpoints=1, ncol=legend_ncol, bbox_to_anchor=(0.5, 0.81), fontsize=16)
    
    # legend title wiht some additional info,   
    l.get_title().set_fontsize('20')
    perc_of_used_features = f"({np.round(variablefeatures/encoded_img_batch.shape[1]*100, 1)}%)"
    l.set_title(f'{features_to_plot.shape[0]} images, each with {featurenumber} features, from which {variablefeatures} are different on at least one image {perc_of_used_features}') 
    
    
    
    # .............................................................................
    # Add, Image examples on the right side of the Plot, 

    
    # create new axis on a plot,

    #.. squeeze the left figure, to addapt the plot to second grid,
    g.gs.update(left=0.05, right=0.45) 

    #.. create new figure on a gridspace, on the right,  
    gs2 = matplotlib.gridspec.GridSpec(1,1, left=0.45)

    # create empty axes within this new gridspec - not sure if that part is required, but it didt work without, 
    ax2 = g.fig.add_subplot(gs2[0], facecolor="white")
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0,1)

    # MUST be added here, because lateron I use coordinates on a final image to calculate text postions, !
    g.fig.subplots_adjust(top=0.8)

    
    
    # find parameters for plotting image examples,

    #.. heatmap locations on g plot
    HeatmapBbox = g.ax_heatmap.get_position() # eg: Bbox([[0.14750679428098784, 0.125], [0.45, 0.7210264900662252]])
    heatmap_intervaly = HeatmapBbox.intervaly[1]-HeatmapBbox.intervaly[0]
    heatmap_top = HeatmapBbox.intervaly[1]
    heatmap_bottom = HeatmapBbox.intervaly[0]
    heatmap_left = HeatmapBbox.intervalx[0]
    heatmap_right = HeatmapBbox.intervalx[1]

    #.. find how much distance to add with each leaf on dendrogram, 
    dedrogram_step_per_leaf = heatmap_intervaly/len(ordered_row_colors)

    
    # .............................................................................
    # plot image examples,
    for x_id, xdim in enumerate(np.linspace(0.5, 0.8, 6)[0:selected_img_examples_to_plot.shape[0]]):
        for y_id, ydim in enumerate((np.linspace(0.15, 0.6, max_nr_of_examples_from_each_class_to_display))[::-1]):
            
            # just in case, there were less pictures then requested in a given class, 
            if np.isnan(selected_img_examples_to_plot.iloc[x_id, y_id]):
                pass
            
            else:
                # 1. subplots with image examples form up to 6 classes,  
                
                # Load one example image, and info on colors and class names, and position in the batch, 
                img_idx_in_raw_img_batch = int(selected_img_examples_to_plot.iloc[x_id, y_id])
                class_name = batch_labels.classname[img_idx_in_raw_img_batch]
                class_color = class_colors[class_name] 
                img = raw_img_batch[img_idx_in_raw_img_batch]
                img_example_name = f"example {y_id+1}"
                
                # Create embded plot with image, on cordinats on g figure, with clustergrid, 
                ax_embd= plt.axes([xdim, ydim, 0.1, 0.1], facecolor=class_color) # facecolor doenst work, 
                ax_embd.imshow(img)
                ax_embd.grid(True)
                ax_embd.set_xticks([])
                ax_embd.set_yticks([])
                #ax_embd.set_title(class_name, color="white")

                # Add rectagle with color corresponding to class arount the image, using coordintas for ax2 only, 
                #. - yes its confusing and it doent work any other way
                new_xdim = (xdim-0.425)/0.45
                rect = plt.Rectangle((new_xdim, 0), 0.115, 0.9, fill=class_color, color=class_color, linewidth=18)
                ax2.add_patch(rect)

                # Add title, to each example image
                font = FontProperties()
                font.set_weight("bold")
                ax_embd.set_title(img_example_name, color="white", fontproperties=font)

                
                
                # 2. Add ticks to heatmap y-axis corresponding to image examples, 
            
                # find how far from the top of the dendrogram img example leaf is,
                position_on_dendrogram = np.where(row_order==img_idx_in_raw_img_batch)[0][0]
                example_label_x_position = heatmap_right
                example_label_y_position = heatmap_top-((position_on_dendrogram+0.5)*dedrogram_step_per_leaf)
                # img_example_label = f"- {class_name}, {y_id+1}" this adds fill clas name a exampel number
                img_example_label = f"{''.join(['-']*(x_id+1+y_id))}example {y_id+1}"
                g.fig.text(example_label_x_position, example_label_y_position, img_example_label, 
                           fontsize=10, ha="left", color=class_color)
           
        
    # .............................................................................
    # Add leaf numbers on dendrogram to help you navigate with image examples, and call them with other functions, 

    # decide on step size, 
    if batch_labels.shape[0]<=50: space_between_nps=5
    if (batch_labels.shape[0]>50) & (batch_labels.shape[0]<=100): space_between_nps=10
    if (batch_labels.shape[0]>100) & (batch_labels.shape[0]<=400): space_between_nps=20
    if (batch_labels.shape[0]>200) & (batch_labels.shape[0]<=1000): space_between_nps=100
    if (batch_labels.shape[0]>1000): space_between_nps=200

    #.. add text
    number_to_display = 0
    for i in range(1000):
        y_np_pos = heatmap_top - (space_between_nps*i*dedrogram_step_per_leaf)
        number_to_display = f"/ {space_between_nps*i} /"
        if y_np_pos>=(heatmap_bottom-dedrogram_step_per_leaf/2):
            g.fig.text(0.5, y_np_pos, number_to_display, fontsize=10, ha="left", color="black")
        else:
            pass       


    # .............................................................................   
    # add lines dividing clusters provided with augmented batch_labels, 
    """
        done here, because I wanted to use values calulated for text with custome image ticks to add cluster names and description,  
    """
    g.fig.axes[3].hlines(y=0, xmin=0, xmax=features_to_plot.shape[1], 
                         colors=cluster_dividing_line_color, linestyles=cluster_dividing_line_style, lw=cluster_dividing_line_width)
    dendrogram_cluster_ordered = batch_labels.dendrogram_clusters[row_order].values
    dendrogram_cluster_names = np.unique(dendrogram_cluster_ordered).tolist()
    for i, cln in enumerate(dendrogram_cluster_names):
        
        # add line
        the_last_image_in_the_cluster = np.where(dendrogram_cluster_ordered==cln)[0][-1]
        g.fig.axes[3].hlines(y=the_last_image_in_the_cluster+1, xmin=0, xmax=features_to_plot.shape[1], 
                             colors=cluster_dividing_line_color, linestyles=cluster_dividing_line_style, lw=cluster_dividing_line_width)
                             # +1 to draw the line below features from that image, 
        
        # add descrition to the cluster, using generasl coordinates, 
        
        # .. cluster name 
        cln_description_ordered = batch_labels.dendrogram_cluster_description_v2[row_order].values
        if add_cluster_description==True:
            cluster_name = f"Cluster {str(cln)}: {cln_description_ordered[the_last_image_in_the_cluster]}"
        else:
            cluster_name = f"Cluster {str(cln)}"
        cluster_name_x_position = heatmap_left+0.01
        cluster_name_y_position = (heatmap_top-((np.where(dendrogram_cluster_ordered==cln)[0][0])*dedrogram_step_per_leaf))-0.02
    
        font = FontProperties()
        font.set_weight("bold")       
        g.fig.text(
            cluster_name_x_position, 
            cluster_name_y_position, 
            cluster_name,
            fontsize=cluster_name_fontsize, 
            ha="left", 
            color=cluster_name_color)



    # .............................................................................
    # Add info on plot items for the viewer and show the plot, 

    # above image examples, 
    g.fig.text(0.7, heatmap_top+0.05, 
               "Image examples from up to six classes", 
               fontsize=16, ha="center", color="black")

    # above leaf numbers on dendrogram
    g.fig.text(0.5, heatmap_top+0.05, "Image nr\non dendrogram", fontsize=16, ha="center", color="black")

    # All Fig. adjustment for title and fig legend
    plt.show();    


    
    # .............................................................................
    # return clusters to allow plotting more image ecxamples with other functions, 
    dct = { "selected_img_examples_to_plot": selected_img_examples_to_plot,
            "img_examples_to_plot_in_all_classes": img_examples_to_plot}
    
    return dct



  
  
  

  

# Function, ................................................................
def calculate_linkage_for_images_with_extracted_features(*,
    encoded_img_batch, batch_labels, class_colors, 
    select_features=True, use_categorical_features=True
   ):

    """
      # Input
        -----------------------------------------------------------------------------------
      . encoded_img_batch  : numpy array, [<img number>, <feature number>] 
      . batch_labels       : dataframe, created with encode_images_with_tfhubmodule, 
                             from feature_extraction_tools
      . class_colors       : dictionary, 
                             {str <"class_name">: str <"color">}
                             CAUTION: colors and class names must be unique !
        ..
      . use_categorical_features :bool, if, True extracted feauture values will be encoded 
                             as 1,2,3, for (0,1], (1, 2], and >2, values respectively,  
      . select_features    : bool, if True, the heatmap, and dendrogram will be constructed 
                             only using features that were different in at least one image
      # Returns,
      . dict               : with:
      . - basic_stats      : dict, on number of features used, found ans selected 
      . - row_linkage      : np.array with row linkage caulated suong ward alg, 
      . - batch_label      : pd.DataFrame, as in input, but with two additional columns, 
                             - color assigned to each image
                             - image position on the dendrogram, where 0 is on the top
      . - paremeters.      : dict, with use_categorical_features, select_features, class_colors,   
        -----------------------------------------------------------------------------------
      # test the order, 
        just in case somethign doenst work use the following code to compare
        the color order with the one created using function clustered_histogram_with_image_examples
    
        barh = np.array([1]*len(img_order_on_dendrogram))
        barxpos = np.arange(len(img_order_on_dendrogram))
        fig, ax = plt.subplots(figsize=(4,12))
        ax.barh(barxpos, width=barh, color=img_class_colors_in_img_batch[img_order_on_dendrogram[::-1]])
    
    
    """
    
    
    # data preparation,, .......................................................    
        
    # work on copies, 
    encoded_img_batch = encoded_img_batch.copy()
    batch_labels = batch_labels.copy()
        
    # categorise features, into 1,2 and 3, 
    simplefeatures = encoded_img_batch.copy()
    simplefeatures[encoded_img_batch<=1]=1
    simplefeatures[np.logical_and(encoded_img_batch>1, encoded_img_batch<=2)]=2
    simplefeatures[encoded_img_batch>2]=3

    # find variable features 
    feature_sums = simplefeatures.sum(axis=0)
    selector = np.logical_or(
                    feature_sums==1*simplefeatures.shape[0], 
                    feature_sums==2*simplefeatures.shape[0],
                    feature_sums==3*simplefeatures.shape[0],
                )==False

    # use raw or simplified/sharpenned features to create heatmap
    if use_categorical_features==False:
        features_to_plot = encoded_img_batch # features_to_plot - i kept the name from other function, 
    else:
        features_to_plot = simplefeatures

    # remove all features that are uniform across all encoded images, 
    if select_features==True:
        features_to_plot = features_to_plot[:, selector] 
    else:
        pass    

    
    
    # calulate or extract data to return, .......................................................
    
    # stats
    basic_stats = {
        "total_feature_number" : encoded_img_batch.shape[1], 
        "variable_feature_number" : selector.sum(),
        "number_of_features_used_for_hclus" : features_to_plot.shape[1] 
    }    
    
    # chierarchical clustering on x/y axses
    correlations_array = np.asarray(features_to_plot)
    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='ward')
    img_order_on_dendrogram = leaves_list(row_linkage) # ie, the order of samples using img_order_in_batch as value,  

    # img position in the cluster, - where 0 is on the top,
    "its an, index that you need to use to get image examples, in the order given on a dendrogram"
    img_position_on_dendrogram =  pd.Series(pd.DataFrame(img_order_on_dendrogram).sort_values(0).index.values)
    
    # assign colors to class names, 
    color_mix = dict(zip(list(class_colors.keys()), list(class_colors.values()))) # because all my plots must be pretty and with the ssame colors !
    img_class_colors_in_img_batch    = pd.Series(batch_labels.classname.values.tolist()).map(color_mix) 
    
    # add data to batch_labels table
    batch_labels["img_position_on_dendrogram"] = img_position_on_dendrogram
    batch_labels["img_class_colors_in_img_batch"] = img_class_colors_in_img_batch

    # dict to rreturn
    dict_to_return = {
        "info":"img order allow recreates dendrogram from top to bottom, using img indexes, img position shows whre a given image resides on the dendrogram",
        "stats": basic_stats,
        "batch_labels":batch_labels,
        "img_order_on_dedrogram":img_order_on_dendrogram,
        "row_linkage": row_linkage,
        "parameters":{
            "class_colors": class_colors,
            "use_categorical_features":use_categorical_features,
            "select_features": select_features
        }
    }
    
    return dict_to_return
    
    
    
    
    
    
    
    
# Function, ..........................................................................................................
def add_descriptive_notes_to_each_cluster_in_batch_labels(*, batch_labels):
    """
        small fucntions used by find_clusters_on_dendrogram(), 
        to provide descritive names for clusters identified by that function
        using original class names priovided with labelled data, 
        
        adds three new columns to batch_labels, described in find_clusters_on_dendrogram()
        called, endrogram_cluster_name/compositions/description
    """

    # work on copy,
    batch_labels = batch_labels.copy()

    # add empty columns to df, 
    empty_row_str = np.zeros(batch_labels.shape[0], dtype=str)
    batch_labels["dendrogram_cluster_name"]= empty_row_str
    batch_labels["dendrogram_cluster_description"]= empty_row_str
    batch_labels["dendrogram_cluster_composition"]= empty_row_str
    batch_labels["dendrogram_cluster_description_v2"]= empty_row_str

    # get cluster names,
    cluster_names = batch_labels.dendrogram_clusters.unique().tolist()

    # construct the cluster name, using clasname
    for cln in cluster_names:
        
        # data preparation, 
        class_counts = batch_labels.classname.loc[batch_labels.dendrogram_clusters==cln].value_counts(normalize=True)
        class_counts_number = batch_labels.classname.loc[batch_labels.dendrogram_clusters==cln].value_counts()
        class_outside_cluster = batch_labels.classname.loc[batch_labels.dendrogram_clusters!=cln].values
        
        # dendrogram_cluster_name - after the most frequent class
        batch_labels.loc[batch_labels.dendrogram_clusters==cln, "dendrogram_cluster_name"]=list(class_counts.index)[0]
        
        # dendrogram_cluster_composition
        number_of_class_examples_outside_cluster = (class_outside_cluster==list(class_counts.index)[0]).sum()
        perc_of_class_exaples_in_that_cluster = class_counts_number[0]/(number_of_class_examples_outside_cluster+class_counts_number[0])*100
        dendrogram_cluster_composition = f"cluster contains {np.round(perc_of_class_exaples_in_that_cluster, 1)}% of all images with {list(class_counts.index)[0]} in dataset"
        batch_labels.loc[batch_labels.dendrogram_clusters==cln, "dendrogram_cluster_composition"]=dendrogram_cluster_composition
        
        # add descriptive information on the cluster composition
        if class_counts[0]>=0.51:   
            dendrogram_cluster_description=f"{np.round(class_counts[0],3)*100}% of images in that cluster shows {list(class_counts.index)[0]}"
            
        if class_counts[0]<0.51:   
            dendrogram_cluster_description=f"{np.round(class_counts[0],3)*100}% of images in that cluster shows {list(class_counts.index)[0]}, and {np.round(class_counts[1],3)*100}% {list(class_counts.index)[1]} + ..."
        batch_labels.loc[batch_labels.dendrogram_clusters==cln, "dendrogram_cluster_description"]=dendrogram_cluster_description

        # dendrogram_cluster_description_v2
        class_counts = pd.DataFrame(class_counts)
        class_counts = class_counts[0:3]
        class_counts.reset_index(drop=False, inplace=True)
        class_counts.columns = ["classname", "perc"]
        class_counts.perc = [f'{str(np.round(x*100,1))}%' for x in class_counts.perc.values.tolist()]
        class_counts["name_number"]= [": "]*class_counts.shape[0]
        class_counts["end_class"]= [", "]*class_counts.shape[0]
        class_counts = class_counts.loc[:,["classname", "name_number", "perc","end_class"]]
        dendrogram_cluster_description_v2 = "".join(class_counts.stack().values.flatten().tolist())
        #...
        batch_labels.loc[batch_labels.dendrogram_clusters==cln, "dendrogram_cluster_description_v2"]=dendrogram_cluster_description_v2
        
    return batch_labels


  
  
  
  
  
  
  

# Function, ..........................................................................................................
def find_clusters_on_dendrogram(*, linkage_and_batchlabels, min_clusters=None, max_clusters=None, verbose=False):
    """
        Function that automatically, find the similar number of clusters on dendrogram to number of classes 
        used for training data it updates and returns batch_labels with new columns, 
        
        Imput
        .......................
        linkage_and_batchlabels   : dict, object returned by calculate_linkage_for_images_with_extracted_features()
        min_clusters              : int, default=None, minimal number of cluster that shodul be found on dendrogram 
                                    if None, it will use number oif keys from class_colors in linkage_and_batchlabels,
        max_clusters              : None, maximal number of cluster that shodul be found on dendrogram 
                                    if None, it will use 1.5*number of keys from class_colors in linkage_and_batchlabels,
        Returns:
        .......................
        Pandas DataFrame          : df, batch_labels from linkage_and_batchlabels, with three additional columns, 
                                    - dendrogram_clusters     : int. from 1:max_found_dendrograms, assigned to each row
                                    - dendrogram_cluster_name : str, after class name of the most common class in that cluster
                                    - dendrogram_cluster_composition: str, descriptive, describes how many items belowns to one 
                                      or two most common classes in that cluster
                                    - dendrogram_cluster_description: str, descriptive, descirbes how many istems in entire dataset, from the class 
                                      used to name each cluster, can be found in these clusters, eg: 80% indicates that only 80% of bikes, 
                                      items of a class bike, is inside the dendrogram cluster named bikes, 
    """    

    # prepare max and min expected clusters, that we wish to find on dendrogram, 
    if  min_clusters==None:
        min_clusters = len(linkage_and_batchlabels["parameters"]["class_colors"].keys()) 
    if  max_clusters==None:
        max_clusters = len(linkage_and_batchlabels["parameters"]["class_colors"].keys())+int(len(linkage_and_batchlabels["parameters"]["class_colors"].keys())*0.5)
        
    # data preparation, .......................................
    
    # extract
    row_linkage = linkage_and_batchlabels['row_linkage'].copy()
    batch_labels = linkage_and_batchlabels['batch_labels'].copy()
    
    # find cluster number with 100 different cutoff threshold on dendrogram, 
    """
        Important, with large datasets, the cutoff, is lower then with small datasets, to get the same number of clusters, 
        From that reason, you must increate the number of steps and lower start and end points, 
        This part of code could be rewritten, to break if the searched coditions are met.
        For now, I added start, end and step manually, due to lack of time, 
    """
    # ...
    cutoff_start=0.001
    cutoff_end  =0.9
    cutoff_step =(row_linkage.shape[0]+1)*20
    # ...
    cluster_number = list()
    Threshold_values = list()
    # ...
    for i in np.linspace(cutoff_start, cutoff_end, cutoff_step):
        Threshold_values.append(i)
        dendrogram_clusters = sch.fcluster(row_linkage, i*row_linkage.max(), 'distance')
        cluster_number.append(np.unique(dendrogram_clusters).shape[0])
        if np.unique(dendrogram_clusters).shape[0]<min_clusters:
            break
            # then this will be the max cluster that will be used in case other options are not available
    # ...
    Threshold_values = np.array(Threshold_values)
    cluster_number = np.array(cluster_number)

    # decide on the threshold, and cluster number do display
    cutoff_meeting_my_criteria = np.logical_and(cluster_number>=min_clusters, cluster_number<=max_clusters)
    # ...
    if sum(cutoff_meeting_my_criteria)>0:
        # find cutoff that allows getting the same or similar number of clusters as in class_colors
        cutoff_index_to_use = np.where(cutoff_meeting_my_criteria==True)[0][-1]
        
    else:
        # use the first criterion that have min, nr of clusters, and its closes to requested one, given the coditions used
        cutoff_index_to_use = np.where(cluster_number)[0][-1]
    # ...
    dendrogram_clusters = sch.fcluster(row_linkage, Threshold_values[cutoff_index_to_use]*row_linkage.max(), 'distance')   

    # info
    if verbose==True:
         print(f"following clusters were found on dendrogram {np.unique(dendrogram_clusters)} with cutoff {Threshold_values[cutoff_index_to_use]} ")

            
    # add to batch_labels
    batch_labels["dendrogram_clusters"] = dendrogram_clusters            
            
    # add cluster names, and descritive information
    batch_labels = add_descriptive_notes_to_each_cluster_in_batch_labels(batch_labels=batch_labels)
    
    # .......................................
    return batch_labels
  
  
  
  
  
  
  
# Function ............................................................................................................................
def create_clustered_heatmap_with_img_examples(*, 
    raw_img_batch, 
    load_dir, # raw data - one level below dataset_name file folder with class_name folders with images
    extracted_features_load_dir=None,
    logfiles_for_extracted_features_load_dir=None,  # usually, the same as extracted_features_load_dir                                                                
    # ....
    module_names, 
    dataset_name, 
    subset_names, 
    class_colors, 
    verbose=False
):
    """
        .................   ...........................................................................    
        Property            Description
        .................   ...........................................................................    
        
        * Function          This is a Wrapped function for clustered_histogram_with_image_examples() function that draws 
                            clustered heatmap, with image examples from up to six, largest clastuers of images, clustered based on 
                            similarity in features extracted with tfhub modules 
                            ....
                            The function, uses calculate_linkage_for_images_with_extracted_features(), 
                            and find_clusters_on_dendrogram()
                            ....
                            Data are loaded uzsing load_raw_img_batch(), and load_encoded_imgbatch_using_logfile() from data loaders, 
        # Inputs
        .................   ...........................................................................    
        . raw_img_batch     : array with RGB images, [?, pixel size, pixel size, 3]
                              values 0-1, 
                              Caution ! Images must be ordered in the same way as in img_batch labels, 
                                        and arrays wiht extracted features !!!!
        . load_dir          : str, PATH to directory with folder that contains images grouped in subsets, 
                              each subset needs to have 
        . module_names      : list, with strings, each strinf is a unique name given to tfhub module, 
                              used to extract features from dataset, 
        . dataset_name      : name of folder where a t least one datasubset is stored, 
                              together with extracted features, batchfiles and logfiles,                       
        . subset_names      : name of the folder that contains images in directories names as their classes, 
        . class_colors      : dict, key: str or int, label for the legend, value: str, color for the legend
        
        # Returns,
        .................   ...........................................................................  
        . Figure            figure created usng only Matplotlb and Seaborn, basic functions,
        . dictionary        with dict, with 
                            'info'; str, with basic info on the clusterng,
                            'stats'; numb er of used, unused and unique features in each dataset, 
                            'batch_labels'; pandas df, with info on each image in the batch,
                            'img_order_on_dedrogram'; np.array, 
                            'row_linkage'; stats. hiererch, libnkage array for rows (images) 
                            'parameters', dict, oparameters used for dendrogram
                            'hclus_prediction_acc'; accuracy, calulated using the same number of clusters on dendrogram, 
                                                    as the classs on labelled images,
                            'plotted_image_examples', dict, with df's showing idx for images in six largest classes
                                                      and in all classes, if more were available,
    """
    

    # set up paths, 
    
    if extracted_features_load_dir==None:
        extracted_features_load_dir=load_dir
    else:
        pass
        
    if logfiles_for_extracted_features_load_dir==None:
        logfiles_for_extracted_features_load_dir=extracted_features_load_dir=load_dir
    else:
        pass


    # collect log files for encoded images
    try:
        os.chdir(load_dir)
    except:
        if verbose==True:
            print(f"Error: {load_dir} NOT FOUND")
        else:
            pass


    # load raw and ecoded img batches, labels and join different batches into one, in the same order and in batch labels, then calulate distances and plot the heatmap
    results_with_each_module = dict() 
    for module_name in module_names:

        
        # find all logfiles that were created for a given dataset_name & module_name
        "the pattern must be exaclty as below, other wise eg resnet and resnet1 woudl be found with resnet module name"
        try:
            os.chdir(logfiles_for_extracted_features_load_dir)
        except:
            if verbose==True:
                print(f"Error: {logfiles_for_extracted_features_load_dir} NOT FOUND")
            else:
                pass
        logfiles = []
        for file in glob.glob(f"{''.join([module_name,'_',dataset_name])}*_logfile.csv"):
            logfiles.append(file)

            
        #  chech if you have only one log-file per combination - if not, there is a problem,
        if len(logfiles)==0:
            if verbose==True:
                print(f"KeyError: logfile with {module_name} was not found in searched directory")
        if len(logfiles)>1:
            if verbose==True:
                print(f"KeyError: {module_name} matches more then one directory, please make sure the names are unique")

        if len(logfiles)==1:
            try:
                os.chdir(extracted_features_load_dir)
            except:
                if verbose==True:
                    print(f"Error: {extracted_features_load_dir} NOT FOUND")
                else:
                    pass
                
            # load encoded img, and batch labels,  
            encoded_img_batch, batch_labels = load_encoded_imgbatch_using_logfile(
                                                logfile_name=logfiles[0], 
                                                load_datasetnames=subset_names ,
                                                verbose=verbose)


            # caslulate and collect data on each dendrogram,
            results_with_each_module[module_name] = calculate_linkage_for_images_with_extracted_features(
                encoded_img_batch=encoded_img_batch, 
                batch_labels=batch_labels, 
                class_colors=class_colors
            )

            # find clusters on dendrogram and update batch labels with descriptive labels, 
            results_with_each_module[module_name]["batch_labels"] = find_clusters_on_dendrogram(
                linkage_and_batchlabels=results_with_each_module[module_name], 
                 min_clusters=len(list(class_colors.keys())), 
                 max_clusters=len(list(class_colors.keys()))+2,
                verbose=verbose
            ) 

            #.. calulate accuracy for cluster prediction with hierachical clustering, 
            acc = accuracy_score(results_with_each_module[module_name]["batch_labels"].classname, results_with_each_module[module_name]["batch_labels"].dendrogram_cluster_name)
            results_with_each_module[module_name]["hclus_prediction_acc"]=acc

            # plot clustered heatmap with image examples,  
            results_with_each_module[module_name]["plotted_image_examples"] = clustered_histogram_with_image_examples(
                encoded_img_batch = encoded_img_batch,
                batch_labels = results_with_each_module[module_name]["batch_labels"],
                raw_img_batch = raw_img_batch,
                row_linkage=results_with_each_module[module_name]["row_linkage"], 
                plot_title = f"Features extracted from {', '.join(subset_names)} set, with {module_name}, allow hclust with acc={np.round(acc, 2)}%",
                method_name = f" ({module_name})",
                class_colors = class_colors,
                verbose=verbose
                )

    # ...                                        
    return results_with_each_module







