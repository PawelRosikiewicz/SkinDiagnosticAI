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
import matplotlib.pyplot as plt # for making plots, 

from PIL import Image, ImageDraw
import matplotlib.gridspec
from matplotlib.font_manager import FontProperties



# Function, ..................................................................................

def create_class_colors_dict(*, list_of_unique_names, cmap_name="tab20", cmap_colors_from=0, cmap_colors_to=1):
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





