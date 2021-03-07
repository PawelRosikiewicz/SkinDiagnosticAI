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






# Function, .........................................................................................
def create_spaces_between_img_clusters(*, df_list, scaling_for_cluster_space, verbose=False):
    """
        Helper function for plot_img_examples(), 
        adds, spaces between groups of images that will be plotted, 
    """
    
    # calulate new img widht and space between clusters of images, 
    img_width = df_list[0].img_width[0] # I am assuming all images have the same width,
    cols_with_images = np.array([x.img_x_position.unique().size for x in  df_list]).sum()
    group_nr = len(df_list)

    # ...
    new_img_width = (img_width*cols_with_images)/(cols_with_images+((group_nr-1)*scaling_for_cluster_space))
    space_between_clusters = new_img_width*scaling_for_cluster_space

    # adjust sizes, 
    global_posx=0
    for i in range(len(df_list)):

        cl_df = df_list[i].copy()

        # update img widths in df, 
        cl_df.loc[:,"img_width"] = new_img_width

        # update start of each image,
        old_img_posx_from_cl_df = cl_df.loc[:,"img_x_position"].unique()
        
        if verbose==True: 
            print(old_img_posx_from_cl_df)
        else:
            pass

        # ...
        new_img_posx = []
        for img_col_nr in range(len(old_img_posx_from_cl_df)):        
            new_img_posx.append(global_posx)
            # update global_posx, with new_img_width
            global_posx+=new_img_width

        # update new_img_width with space between images
        global_posx+=space_between_clusters

        # fill in df with updatesd values
        for j, new_posx in enumerate(new_img_posx):
            cl_df.loc[cl_df.img_x_position==old_img_posx_from_cl_df[j],"img_x_position"]=new_img_posx[j]
        
        if verbose==True: 
            print(new_img_posx)
        else:
            pass

        # return cl_df into df_list
        df_list[i] = cl_df
    
    return df_list


  
  

# Function, .........................................................................................

def plot_img_examples(*, selected_img_batch, img_name=None, img_groupname=None, img_color=None, class_colors_for_legend=None, title=None, fig_facecolor="white", 
                     space_between_clusters=0.35, figsize_scaling = 2, max_img_per_col = None, subplots_adjust_top = None, legend_loc="center left", verbose=False):

    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Plots RGB imgage from array [?, pixel_size, pixel_size, 3]
                            images can be labelled, with label displayed as color box arround each image, and 
                            grouped, where the images belonging to different groups are separated from each other 
                            Images are plotted in the same order as in the array, except for the gorups, whther order of
                            images in array is maintained, but each group is plotted separately, in alphabetical order.
                            color labels/classes are described in legend.
                            # ...
                            all imagess are plotted with Matplotlib.pyplot.imshow
                            labels on images are added using OpneCV library, 
                            # ...
                            Caution: the order of inputs in img_groupname, img_color, and  img_name must correspond 
                                     to the order of images in selected_img_batch
                            
                            # ....
                            the images shoudl be displayed as follow:

                                 legend : translates colors into class names,

                            one group of images./each image has a name, where you can put your description,   
                            |                  /
                            ..  . ... .. ... ..
                            ..  . ... .  ... .  - each group may have images from different classes, 
                            .     ..  .  ... .    and these classes may be, color box shows the class of the image

        # Inputs
        .......................     ...........................................................................
        . selected_img_batch        : array, [?, pixel_size, pixel_size, 3], with RGB images, 
                                      values 0-1, REQUIRED for figure, 
        . ...
        . img_groupname             : list, with strings or int, 
                                      used to create groups of images displayed separately, will be displayed above each group, 
                                      it is not connected to img_color, but, can be, if the same value will be provided to both, 
        . img_color                 : list, with strings, used to create color box arround each image, 
        . img_name                  : list, with strings or int, will be displayed above each image, 
        . title                     : str, title above the figure, with all images, 
        . fig_facecolor             : str, figure facecolor, default="lightgrey"
        . ....
        . space_between_clusters    : float or int,  >=0, default=0.35, the space between groups of images, 
                                      value is the fraction of whe width of one image example plotted in any group,
        . figsize_scaling           : int >=1, default=3, how large the figure shoudl be, as multiplication of row/col nr's
        . max_img_per_col           : int >=1, how many image examples shodul be plotted in one column, 
        . subplots_adjust_top       : float (0,1), default=0.85,  how much space to use for image example, 
                                      the rest is used for legen,d, title and group labels, if any,
        . verbose                   : bool, default=False
        . legend_loc                : str, for description see matplotlib.pyplot.legend, loc parameter
        . class_colors_for_legend   : dict, key: str or int, label for the legend, value: str, color for the legend. 
                                     {str <"class_name">: str <"color">}
                                     CAUTION: colors and class names must be unique !
        
        # Returns
        .......................     ...........................................................................
        Matplotlib figure, 

    """


    
    # some info for the user
    if verbose==True and  selected_img_batch.shape[0]>500:
        print("CAUTION; you are plotting over 500 iages with my function, this may take time, ...")
        print("         you may alsso try using smaller number of images to make the plot faster")
    else: 
        pass

    
    # STEP 1. DATA PREPARATION, 
    
    # ......................................
    # set optional parameters, 
    if img_name==None: 
        img_name =  [" ".join(["img",str(x)]) for x in list(range(selected_img_batch.shape[0]))]
        if verbose==True: 
            print("img_name were not specified ...  generic names such as img 1,2,3, will be used ")
        else: 
            pass
    else: 
        pass
    
    # ...
    if img_color==None: 
        img_color =  ["darkgrey"]*selected_img_batch.shape[0]
        if verbose==True: 
            print("img_color were not specified ...  we will use one color - darkgrey for all images")
        else: 
            pass
    else:
        pass
    
    # ...
    if img_groupname==None: 
        img_groupname =  ["Images in dataset"]*selected_img_batch.shape[0]
        if verbose==True: 
            print("img_groupname were not specified ...  all images will be plotted one after anothe, as they woudl belong to one group, cluster, ...")
        else: 
            pass
    else: 
        pass
    
    # ...
    if class_colors_for_legend==None:     
        if verbose==True: 
            print("""class_colors_for_legend were not specified ... they are rerquired for the legend, to connect color boxes arround figures 
                  with labels/classes, make sure you provided these values in dictionary (key=img class/label, value=color)""")
        else: 
            pass
    else:
        pass
        
        
    # ...
    if max_img_per_col==None:
        if selected_img_batch.shape[0]<=200:
            max_img_per_col = 10
        else:
            max_img_per_col = int(np.ceil(selected_img_batch.shape[0]/20))
    
    # ...
    title_fonsize = 18*np.log(max_img_per_col)*(figsize_scaling/2)
    
                                         
    #.. create df, for easier work, 
    info_on_selected_images = pd.DataFrame(list(zip(img_name, img_groupname, img_color, list(range(len(img_name))))), 
                                           columns=("img_name", "img_groupname", "img_color", "img_ID"))
                                           # img, ID is added to use idx in img_batch array despite having dasta dividded into smaller df's for plot

    #.. find groups of pictures that will be plotted separately on the same figure, 
    img_groups_to_plot = info_on_selected_images.img_groupname.unique()
    nr_of_images_per_group = info_on_selected_images.groupby("img_groupname").count().img_name.values
    nr_of_columns_with_images_per_group = np.array([int(np.ceil(x/max_img_per_col)) for x in nr_of_images_per_group.tolist()])

    
    # ......................................
    # calulate how much space to leave for title, group names, and legend
    if subplots_adjust_top == None:
        if len(img_groups_to_plot.tolist())==1:
            subplots_adjust_top = 0.85
        else:
            subplots_adjust_top = 0.8
    else:
        pass    

    
    # ......................................
    # Set dimensions of individual images,
    #.    (caution, these values may be adjuster later on, for spcaes between groups)

    #.. 1) rows: find row number
    if nr_of_columns_with_images_per_group.max()>1:
        total_nr_of_rows = int(max_img_per_col)
    else:
        total_nr_of_rows = int(nr_of_images_per_group.max())

    #.. 2) Calulate height and width of each image, on fig, coordinates, 
    one_img_width = 1/(nr_of_columns_with_images_per_group).sum()
    one_img_height = (subplots_adjust_top/total_nr_of_rows)
    
    
    # ......................................
    # Prepare small df, with info on images and x,y positions of their lower-left corners, - treat each cluster separately, 
    "here I do not add spaces between columns separating groups of images - its done later on with helper function to gain more control"

    plot_info_df_list = list()
    for group_i, group_name in enumerate(img_groups_to_plot.tolist()):

        # extrad data on images belonging to that particular cluster
        row_filter = info_on_selected_images.img_groupname==group_name
        group_info_on_selected_images = pd.DataFrame(info_on_selected_images.loc[row_filter,:])
        group_info_on_selected_images = group_info_on_selected_images.reset_index(drop=True)# because otherwise I saw SettingWithCopyWarning,

        # select columns on the image that will be filled with selectecd images from a given cluster, 
        if group_i==0:
            group_assigned_columns_on_figure = np.arange(0,nr_of_columns_with_images_per_group[ group_i])
        else:
            columns_before = nr_of_columns_with_images_per_group[0: group_i].sum()
            group_assigned_columns_on_figure = np.arange(0,nr_of_columns_with_images_per_group[group_i])+columns_before


        # caulate x,y - lower left corner coordinates for each image in each group in each column, 
        group_image_x_position=[] # values from 0 to 1
        group_image_y_position=[] # values goest from 1 to 0, as pictures are starting from the top, lower-left corner, 
        # ...
        column = 0 # it helps moving over 
        img_counter_per_column = 0 # restarted each time a max_img_per_col is reached, 
        # ...
        print(nr_of_images_per_group[group_i])
        for i in range(nr_of_images_per_group[group_i]): # removed np.arange.
            img_counter_per_column+=1 
            if img_counter_per_column>max_img_per_col:
                img_counter_per_column=1
                column+=1              
            # ..
            group_image_x_position.append(group_assigned_columns_on_figure[column]*one_img_width)
            group_image_y_position.append((1-img_counter_per_column*one_img_height)-(1-subplots_adjust_top))

        # Add these into the group_info_on_selected_images
        group_info_on_selected_images["img_width"] = [one_img_width]*(group_info_on_selected_images.shape[0])
        group_info_on_selected_images["img_height"] = [one_img_height]*(group_info_on_selected_images.shape[0])
        group_info_on_selected_images["img_x_position"] = pd.Series(group_image_x_position)
        group_info_on_selected_images["img_y_position"] = pd.Series(group_image_y_position)

        # and place in the list, for plotting later on, 
        group_info_on_selected_images = group_info_on_selected_images.reset_index(drop=True)# just in case :)
        plot_info_df_list.append(group_info_on_selected_images)    


    #print("DONE .....")#########################################
    print(plot_info_df_list)#########################################
    
    # ......................................
    # add spaces bewtween clusters,     
    if len(img_groups_to_plot)>1:
        plot_info_df_list = create_spaces_between_img_clusters(df_list=plot_info_df_list, scaling_for_cluster_space=space_between_clusters)
    else:
        pass

    
    # ......................................
    # create the main figure, with images, title and group description
    mpl.rc('axes',linewidth=5)
    fig = plt.figure(figsize=( (nr_of_columns_with_images_per_group).sum()*figsize_scaling*subplots_adjust_top, total_nr_of_rows*figsize_scaling*0.8), 
                     facecolor=fig_facecolor)

    if title!=None:
        fig.suptitle(title, fontsize=title_fonsize, ha="center", color="black")
    else:
        pass

    # add images in each group
    for i, cluster_df in enumerate(plot_info_df_list):

        
        # add group name above each img group, and some lines for cosmetics, :)
        if img_groupname!=None:
            groupname = f"{cluster_df.img_groupname[0]}"
            # ...
            imggroup_x_from = cluster_df.img_x_position.min()
            imggroup_x_to = cluster_df.img_x_position.max()+cluster_df.img_width[0]
            # ...
            if len(plot_info_df_list)>1:
                groupname_ha="left"
                if len(groupname)<=10: 
                    groupname_rotation=45
                    groupname_fontsize=title_fonsize*0.5
                    groupname_ypos = subplots_adjust_top+subplots_adjust_top/12
                if len(groupname)>10:
                    groupname_rotation=20
                    groupname_fontsize=title_fonsize*0.4
                    groupname_ypos = subplots_adjust_top+subplots_adjust_top/12
            else:
                groupname_rotation=0
                groupname_ha="center"
                groupname_fontsize=title_fonsize*0.8
                groupname_ypos = subplots_adjust_top+subplots_adjust_top/(title_fonsize*2)

            fig.text((imggroup_x_to-imggroup_x_from)/2+imggroup_x_from, groupname_ypos, groupname, 
                     fontsize=groupname_fontsize, rotation=groupname_rotation, ha=groupname_ha)
        else:
            pass
            
        # add line above each cluster - looks nice
        axline = plt.axes([imggroup_x_from, 
                           subplots_adjust_top+subplots_adjust_top/300, 
                           imggroup_x_to-imggroup_x_from,
                           subplots_adjust_top/300
                          ], facecolor="black")        
        axline.grid(False)
        axline.set_xticks([])
        axline.set_yticks([])      
        axline.spines["right"].set_visible(False) # and below, remove white border, 
        axline.spines["left"].set_visible(False)
        axline.spines["top"].set_visible(False)
        axline.spines["bottom"].set_visible(False)        

        # plot images, 
        for img_nr in range(cluster_df.shape[0]):

            # get data,
            img = selected_img_batch[int(cluster_df.img_ID[img_nr])]
            img_color = cluster_df.img_color[img_nr]
            # ...
            xpos = cluster_df.img_x_position[img_nr] 
            ypos = cluster_df.img_y_position[img_nr]
            imgw = cluster_df.img_width[img_nr]
            imgh = cluster_df.img_height[img_nr]

            
            # add text to img with its label, - Caution it will modify selected_img_batch !
            
            # because not everyone has OpenCV library installed:
            try:
                text_y_pos = int(img.shape[1]/7) # from the top, 
                cv2.putText(img, text=f"{cluster_df.img_name[img_nr]}", 
                    org=(5,text_y_pos),  # in pixels, x (from left to right), y (from top, to botom)
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX,  # required, 
                    fontScale=1, 
                    color=mpl.colors.to_rgb("white"),
                    thickness=2, lineType=cv2.LINE_AA)
            except:
                if verbose==True:
                    print("img_names coulnt be added to each image example")
                    print("please check if you have OpneCV library installed in your encviroment")
                else:
                    pass

            # create the box arround each image indicating its label (class assigment, if any)
            space_for_color_box = subplots_adjust_top/300
            # ...
            ax = plt.axes([xpos, ypos, imgw, imgh], facecolor=img_color)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # add image withint that box
            ax2 = plt.axes([xpos+space_for_color_box/2, 
                           ypos+space_for_color_box/2, 
                           (imgw-space_for_color_box), 
                           (imgh-space_for_color_box)
                          ], facecolor=img_color)
            ax2.grid(False)
            ax2.imshow(img)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.spines["right"].set_visible(False) # and below, remove white border, 
            ax2.spines["left"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)


    # .............................................................................
    # Main Legend - later on, I wish to modify that part, 
    if class_colors_for_legend!=None:
        # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
        patch_list_for_legend =[]
        count_items = 0
        for i, cl_name in enumerate(list(class_colors_for_legend.keys())):
            cl_color = class_colors_for_legend[cl_name]
            patch_list_for_legend.append(mpatches.Patch(color=cl_color, label=cl_name))
            
        # legend central position, 
        l_bbox = (0.6, subplots_adjust_top+0.6*(1-subplots_adjust_top)) 
                  # ie just above the middle of the space between group name and the title, 
            
        # number of columns for legend labels to display, 
        legend_ncol = int(np.ceil(len(list(class_colors_for_legend.keys()))/2))
            
        # add patches to plot,
        l = fig.legend(handles=patch_list_for_legend,
            frameon=False, 
            scatterpoints=1, ncol=6, 
            loc=legend_loc,
            bbox_to_anchor=l_bbox, 
            fontsize=title_fonsize*0.5)

        # legend title wiht some additional info,   
        # l.get_title().set_fontsize(str(int(title_fonsize*0.4)))
        # l.set_title(f'legend title') 
    else:
        if verbose==True: 
            print("Caution: class_colors_for_legend, werer not provided, this dict. is required for plotting the legend, see more in function help")
        else:
            pass
        
    
    plt.show();
    
    
    
    
    
    
    
# Function, .........................................................................................

def plot_img_examples_from_dendrogram(*, raw_img_batch, data_for_plot, plot_title="", figsize_scaling=2, space_between_clusters=0.5, 
                                      number_of_img_examples=100, plot_img_from=None, 
                                      plot_img_to=None, legend_loc="center", 
                                      max_img_per_col=None):    
    """
        A wrapper function, for plot_img_examples
    """
    
    # set idx 
    if plot_img_from!=None and plot_img_to!=None:
        img_idx = data_for_plot['img_order_on_dedrogram'][plot_img_from:plot_img_to].tolist()
        img_names = [" ".join(["image",str(x)]) for x in list(range(plot_img_from, plot_img_to))]
        
    else:
        if number_of_img_examples<raw_img_batch.shape[0]:
          temp = np.unique(np.floor(np.linspace(0,raw_img_batch.shape[0], number_of_img_examples, endpoint=False)).astype(int))
        else:
          temp = np.arange(raw_img_batch.shape[0], dtype="int")
        # ...
        img_idx = data_for_plot['img_order_on_dedrogram'][temp]
        img_names = [" ".join(["image",str(x)]) for x in list(img_idx)]          
          
          
    # find idx if images in batch_labels, but iordered as on dendrogram, 
    selected_df_for_plot = data_for_plot['batch_labels'].loc[img_idx,:]
    selected_df_for_plot.reset_index(drop=False, inplace=True)

    # select and re-order images in the batch for the fucntion, 
    img_idx_in_img_batch = selected_df_for_plot.loc[:,"index"].values
    selected_img_batch = raw_img_batch[selected_df_for_plot.loc[:,"index"].values]

    # plot image examples with clusters as on 
    plot_img_examples(
        selected_img_batch        = selected_img_batch,
        img_groupname             = [" ".join(["Cluster",str(x)]) for x in selected_df_for_plot.loc[:, "dendrogram_clusters"].values.tolist()],
        img_name                  = img_names,
        img_color                 = selected_df_for_plot.loc[:, "img_class_colors_in_img_batch"].values.tolist(),
        class_colors_for_legend   = data_for_plot['parameters']['class_colors'],
        title                     = plot_title,
        legend_loc                = legend_loc,
        max_img_per_col           = max_img_per_col,
        figsize_scaling           = figsize_scaling,
        space_between_clusters    = space_between_clusters
    ) 

