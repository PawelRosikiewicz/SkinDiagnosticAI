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
import tensorflow_hub as hub

import tensorflow as tf # tf.__version__ 
import tensorflow.keras as keras 
import matplotlib.pyplot as plt # for making plots, 
import scipy.stats as stats  # library for statistics and technical programming, 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw
from IPython.display import display
from tensorflow.keras import backend as K # used for housekeeping of tf models,









# Function, ........................................................................
def make_keyword_list(input_item):
    '''
        helper function that turns, lists with str, and enbded list, 
        mixed together, into one flatten list of strings,  
        tranforms, floats and integers into string, 
    '''
    keyword_list = []
    
    if isinstance(input_item, str):
        keyword_list.append(input_item)
    elif isinstance(input_item, int) or isinstance(input_item, float):
        keyword_list.append(str(input_item))
    elif isinstance(input_item, list):
        for item in input_item:
            if isinstance(item, str):
                keyword_list.append(item)
            elif isinstance(item, int) or isinstance(item, float):
                keyword_list.append(str(input_item))
            elif isinstance(item, list):            
                keyword_list.extend(item)
    
    return keyword_list
        


    
# Function, ........................................................................  
def find_strings_mathing_any_pattern(*, input_list=None, pattern_list=None, match_any=True, verbose=False):
    '''
       helper function that will return list wiht items from input list
       that match pattern in at least one item in the pattern_list
       
       match_any      : bool, if True, function returns items that match any patter in pattern list
                         if False, it will return only the list matchin all the patterns in all files in the list
       
    '''
    if input_list is None:
        if verbose==True:
            print(f"No data provided")
        else:
            pass
        return None
    
    else:
        # tuns string and embded lists into one flat list
        input_list = make_keyword_list(input_list)
        
        # turn input list into pd series to allow using pandas str functions, 
        input_series = pd.Series(input_list)
        
        if pattern_list is not None:    
            # tuns string and embded lists into one flat list            
            pattern_list = make_keyword_list(pattern_list)

            # find any item in input items that match any pattern pattern 
            for i, key in enumerate(pattern_list):
                if i==0:
                    resdf = input_series.str.contains(key)
                else:
                    resdf = pd.concat([resdf, input_series.str.contains(key)], axis=1)

            # turn 
            if isinstance(resdf, pd.core.series.Series):
                result_list = input_series.loc[(resdf==True).values.tolist()].values.tolist()
                
            else:
                if match_any==True:
                    result_list = input_series.loc[(resdf.sum(axis=1)>0).values.tolist()].values.tolist()
                else:
                    result_list = input_series.loc[(resdf.sum(axis=1)==resdf.shape[1]).values.tolist()].values.tolist()
        else:
            result_list = input_list
            
        if verbose==True:
            print(f"Provided {len(input_list)} items and {len(pattern_list)} possible patterns to match in each file")
            print(f"Returned {len(result_list)} items that matched to at least one pattern from pattern list")
        else:
            pass
    
    return  result_list    
    

    
    
    
# Function, ........................................................................  
def collect_results(*,
    paths,  # str, or list wiht str,
    filename_keywords, # required, at least one. 
    dirname_keywords=None, 
    filename_match_any=False,                
    dirname_match_any=True, 
    verbose=False
):
    """
        Helper function that will load csv files, in a given file

            paths                : str, or list wiht str,
            filename_keywords,   : str, or list, or list in list, or mixed,  required, at least one. 
            dirname_keywords     : str, or list, or list in list, or mixed,  or None, 
                                   if None, path/s are the final directory where the files will be search, 
                                   if not None, keywords will be used to find selected filenames, 
                                   if "", all files in each provided path/s will be searched for files filename_keywords
            filename_match_any   : bool, def=False, if True, filenames that have at least one pattern provided in 
                                   filename_keywords will be used
                                   if False, only the files that have all keywords in the name will be loaded
            dirname_match_any    : bool, same as filename_match_any, but applied to directory names searched 
                                   within provided path/s
    """

    # set path with results files, 
    path_list = make_keyword_list(paths)
    c=0
    for path in path_list:
        os.chdir(path)

        if dirname_keywords is not None:
            # find and selectct folders inside the path
            dirmane_list=[]
            for dirname in glob.glob("*"):
                dirmane_list.append(dirname)
            selected_dirmane_list = find_strings_mathing_any_pattern(
                input_list   = dirmane_list, 
                pattern_list = dirname_keywords,
                match_any    = dirname_match_any
            )

        else:    
            selected_dirmane_list=[None]

            
        # load all files that match ALL patterns provided wiht filename_keywords
        for dirname in selected_dirmane_list:
            # if no dirmane is selected, it means the path/s are the final destination, 
            if  dirname is not None:
                path_to_file = os.path.join(path, dirname)
            else:
                path_to_file = path            
            os.chdir(path_to_file)
            
            
            # final all files inside using selected patterns such as file extension, 
            filename_list=[]
            for filename in glob.glob("*"):
                filename_list.append(filename)

            selected_filename_list = find_strings_mathing_any_pattern(
                input_list   = filename_list, 
                pattern_list = filename_keywords,
                match_any    = filename_match_any, # returns only the files that contain all provided patterns, 
            )

            for filename in selected_filename_list:
                # load the file and add info on the file name and path to it, 
                one_results_df = pd.read_csv(filename)
                one_results_df["full_path"] =  path_to_file
                one_results_df["filename"] = filename

                # concatenate all the results into one df, 
                if c==0:
                    results_df = one_results_df
                else:
                    results_df = pd.concat([results_df, one_results_df], axis=0)
                    results_df.reset_index(inplace=True, drop=True)
                c+=1    
                
                if verbose==True:
                    print(f"Adding: {filename}")
                    print(f"df shape: {results_df.shape}")
                else:
                    pass

    return results_df
            
