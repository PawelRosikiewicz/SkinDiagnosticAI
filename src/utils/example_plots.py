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
import matplotlib.pyplot as plt # for making plots, 
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import tensorflow as tf
import tensorflow_hub as hub
import scipy.stats as stats  # library for statistics and technical programming, 
import tensorflow.keras as keras  

from PIL import Image, ImageDraw
from IPython.display import display
from tensorflow.keras import backend as K # used for housekeeping of tf models,

import matplotlib.patches as mpatches
from tensorflow.keras.preprocessing.image import ImageDataGenerator




# Function ................................................................................

def plot_example_images_using_generator(generator, title ="", pixel_size=224, class_n_examples=2):
    """
     Plots rgb square size images from generator using Pillow
     
     generator          : kjeras ImageDataGenerator
     pixel_size         : picture size in pixes, as prevoiusly provided to imagegenerator
     class_n_examples   : how many randomly sampled examples of each class to display on a figure
      
      
    """
    
    # extract batch and labels, from a current batch 
    batch_imgs, batch_labels = next(generator) 
    
    # get n_examples from each class
    idx_list = list()
    for ii in range(batch_labels.shape[1]):
        if np.sum(batch_labels[:,ii]==1)>=class_n_examples:
            idx_list.extend(random.sample(np.where(batch_labels[:,ii]==1)[0].tolist(),class_n_examples))
        else:
            pass
    batch_imgs = batch_imgs[idx_list,:]  
        
    # create grid for images from one batch, 
    grid_img = Image.new('RGB', 
        size=(batch_imgs.shape[0]*pixel_size, pixel_size) 
        ) # size: tuple (width, height) in pixels

    # add each image, 
    for j, img in enumerate(batch_imgs):  
        grid_img.paste(
            Image.fromarray(img.astype('uint8')), # important - set as unit8 !
            (j*pixel_size, 0) # position, upper left corner
        )
    
    # info on classes:
    print(f"{title}, Classes: {list(generator.class_indices.keys())}")
    return grid_img 


  
  
  
# Function, ............................................................................
 
def piechart_with_class_composition(*, generator, title="", font_size=10, figsize=(8,6)):    
    """
        Pie chart to diplay class composition in a given keras image generator, 
        with % value on a pie slices and class name and number of items in that class in ticks
        Caution ! it will generate the plot, using only one instance of the generator 
                  with next(generator) !!
        
        Parameters/Input              
        .................   
        * generator        : keras image Generator
        * title            : str, ax.set_title("title")
        * font_size        : int, ticks fontsize
        
        Returns             
        .................   
        * matplotlib       : figure axis object
        * example          : https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html
    """
        
    # extract labels     

    #.. extract the info fro pie chart
    batch_imgs, batch_labels = next(generator)
    batch_labels = pd.DataFrame(batch_labels)
    classlabels = generator.class_indices.keys()

    #.. create pd.series with class names repeated for each image in that batch
    temp = batch_labels.copy().replace(0, np.nan)
    for i, cl in enumerate(classlabels):
        temp.loc[batch_labels.iloc[:,i]==1,i]=cl
    s = temp.unstack().dropna()

    # this part was takes from my dfe package,     
        
    #.. create description for each calls with its percentage in df column
    s = s.value_counts()
    pie_descr = list(s.index)
    data      = [float(x) for x in list(s.values)]
    pie_descr = ["".join([str(int(x))," img's with ",y,
                 " (",str(np.round(x/np.sum(data)*100)),"%)"]) for x,y in zip(data, pie_descr)]

    # the figure
    
    #.. fig, 
    plt.style.use('classic') 
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor="white", figsize=figsize)
    fig.suptitle(f"{title} {temp.shape[0]} images in {len(classlabels)} classes")
    
    #.. plot pie on axis provided from 
    pie_size_scale =0.8
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5*pie_size_scale), radius=pie_size_scale,startangle=-45)
    
    #.. params for widgets
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    kw = dict(arrowprops=dict(arrowstyle="->"),
              bbox=bbox_props, zorder=0, va="center", fontsize=font_size)

    #.. add widgest to pie chart with pie descr
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))*pie_size_scale
        x = np.cos(np.deg2rad(ang))*pie_size_scale
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(pie_descr[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    
    fig.subplots_adjust(top=0.7)
    plt.show()
    
    
    
    

    
# Function, ............................................................................    
    
def classbarplot(*, ax, class_table, subplottitle="", yaxisabel="", classcolors=["grey"]*40):
    """
        Function used by barplot_with_class_composition
        returns axis with barplots, that allow comparing how many images
        is in the same classes in different datasets such as validation and train datasets,
        
        ax            : axis on a figure, 
        class_table   : pd.DataFrame, columns = datasets, rows=classes
                        cells = counts, image number
        
        subplottitle  : str, title for a given subplot, 
        yaxisabel     : str, label for y axis, 
    """
    
    # plot bars, 
    ax.set_title(subplottitle)
    for colnr in range(class_table.shape[1]):
        xaxis_bar_locator = np.array(list(range(0+colnr, class_table.size+colnr+class_table.shape[1]+1, class_table.shape[1]+1)))
        ax.bar(xaxis_bar_locator,class_table.iloc[:,colnr].values, 
               label=class_table.columns[colnr], 
               color=classcolors[colnr], edgecolor=classcolors[colnr])
    #ax.legend(frameon=False)
    ax.set_xlim(-1, class_table.size+class_table.shape[1]*2-1)

    # ticks
    tick_position = np.linspace(0+class_table.shape[1]/2, class_table.size+class_table.shape[1]*2-1-class_table.shape[1]/2, class_table.shape[0])
    ax.set_xticks(tick_position)
    ax.set_xticklabels(list(class_table.index), fontsize=15, color="black", rotation=45, ha="right")

    # Format ticks,
    ax.tick_params(axis='x', colors='black', direction='out', length=4, width=2) # tick only
    ax.tick_params(axis='y', colors='black', direction='out', length=4, width=2) # tick only    
    ax.yaxis.set_ticks_position('left')# shows only that
    ax.xaxis.set_ticks_position('bottom')# shows only that

    # Remove ticks, and axes that you dot'n want, format the other ones,
    ax.spines['top'].set_visible(False) # remove ...
    ax.spines['right'].set_visible(False) # remove ...  
    ax.spines['bottom'].set_linewidth(2) # x axis width
    ax.spines['left'].set_linewidth(2) # y axis width 
    
    # labels
    ax.set_xlabel("Class Name", fontsize=15)
    ax.set_ylabel(yaxisabel, fontsize=15)

    # Add vertical lines from grid,
    ax.yaxis.grid(color='grey', linestyle='--', linewidth=1) # horizontal lines
    
    
  






# Function, ............................................................................

def barplot_with_class_composition(*, generator_dct, title="", font_size=10, figsize=(8,6)):    
    """
        Function created one figure with two plots, 
        showing i) the number, and ii) the % of each class in each dataset
        the datatsest are provided by generators, 
        
        Parameters/Input              
        .................   
        * generator_dct    : dictionary with keras image Generator
                             key : dataset name, value : the generator, 
        * title            : str, ax.set_title("title")
        * font_size        : int, ticks fontsize
        
        Returns             
        .................   
        * matplotlib       : figure axis object
        * example          : https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html
    """
    
    datasetnames = list(generator_dct.keys())
    colors_for_each_dataset = [ "steelblue",  "salmon", "olivedrab", "dimgray", "goldenrod", "salmon"]*100 # just in case
    
    
    # step 1. collect the data on each dataset,
    for nr, setname in enumerate(datasetnames):
        
        # * extract the data form one dataset,
        batch_imgs, batch_labels = next(generator_dct[setname])
        batch_labels = pd.DataFrame(batch_labels)
        classlabels = generator_dct[setname].class_indices.keys()

        # * create pd.series with class names repeated for each image in that batch
        temp = batch_labels.copy().replace(0, np.nan)
        for i, cl in enumerate(classlabels):
            temp.loc[batch_labels.iloc[:,i]==1,i]=cl
        s = temp.unstack().dropna()

        # * create description for each calls with its percentage in df column
        class_count = s.value_counts()
        class_perc = class_count/class_count.sum()*100
        
        # * store
        if nr==0:
            data_class_counts = class_count.copy()
            data_class_perc = class_perc.copy()
        else:
            data_class_counts = pd.concat([data_class_counts, class_count], axis=1)
            data_class_perc = pd.concat([data_class_perc, class_perc], axis=1)
          
    #.. add columns names to not mix datasets lateron, 
    data_class_counts.columns=datasetnames
    data_class_perc.columns=datasetnames
    
    # step 2. Make Barplots,
    fig, axs = plt.subplots(nrows=1, ncols=2, facecolor="white", figsize=figsize)
    fig.suptitle(title, fontsize=20)
    classbarplot(ax=axs[0], class_table=data_class_counts, 
                 subplottitle="Number of Images in each class", yaxisabel="Number", classcolors=colors_for_each_dataset)
    classbarplot(ax=axs[1], class_table=data_class_perc, 
                 subplottitle="Percentage of Images in each class", yaxisabel="Percentage", classcolors=colors_for_each_dataset)
    
    
    # step 3. create patch list for legend,
  
    #.. data for legend, 
    totalclasscount = data_class_counts.sum()

    #.. Add legend to the figure
    patch_list_for_legend =[]
    for i in range(len(datasetnames)):
        one_patch = mpatches.Patch(
            color=colors_for_each_dataset[i], 
            label=f"{datasetnames[i]}: {totalclasscount[datasetnames[i]]} images"
        )
        patch_list_for_legend.append(one_patch)

    l = fig.legend(handles=patch_list_for_legend, 
        loc="center", frameon=False, 
        scatterpoints=1, ncol=3, bbox_to_anchor=(0.5, 0.85), fontsize=12)
    l.set_title("Dataset")
    l.get_title().set_fontsize('15')

    
    # step 4. adjust margins  
    fig.tight_layout()
    fig.subplots_adjust(top=0.75)
    
    plt.show();
    
    
    
    
    
# Function, .............................................................................

def plot_example_images_with_datagenerator(generator, n_batches, image_dim):
    """plot images from generator using Pillow
       1 batch == 1 row of images, 
       . image_dim    : tuple, with two integers eg (299, 299)
       . n_batches     : nr of batchess taken from the file, using the generator
       
    """
    
    # create grid for images from each bath, 
    grid_img = Image.new('RGB', 
                         size=(generator.batch_size*image_dim[0], 
                               n_batches*image_dim[1]) # tuple (width, height) in pixels
                        ) 
    
    # fill in each image on the grid, 
    for i in range(n_batches):
        
        # extract batch and labels, from a current batch 
        batch_imgs, batch_labels = next(generator) 
        
        # Image expects values 0-255, Imshow 0-1
        batch_imgs = batch_imgs * 255
        
        # add each image, 
        for j, img in enumerate(batch_imgs):  
            grid_img.paste(
                Image.fromarray(img.astype('uint8')), # important - set as unit8 !
                (j*image_dim[0], i*image_dim[1]) # position, upper left corner
            )
            
    return grid_img

