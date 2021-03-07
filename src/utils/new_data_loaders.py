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

import graphviz # allows visualizing decision trees,
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier # accepts only numerical data
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



# Function, .............................................................................
def find_different_filetypes(*,
    subsets_dict,
    filetypes_dict,
    path,  
    verbose=False                   
    ):    
    """
        A generaric Function that allows to find files that:
        * are grouped together, eg by subset name like train, test, valid
        * have the same core name, but different affix 
        This function to build new, logfile for data encoding - this aoows adding 
        ASSUMPTION: coprresponding batch labels and extracted features have the same file name
                    except for affix, _encoded.npy and _labels.csv  
        # Inputs:            
        . subsets_dict       : dict, 
                               <key>    : str, name of the group of files eg: test, train, etc..
                               <value>  : str, part of the pattern a apttern that allows to find all files belonging to one group
                               important: in case more then one pattern must be used to identify files form one group,   
                                          just name them with numbers, and later on replace in df returned by the function,
        . filetypes_dict     : dict,
                               <key>    : str, name of affix added to the file of a given type
                               <value>  : str, of affix added to the file of a given type                   
        . path               : full path to directory with fileas searched by the function, 
        . verbose            : bool,              
                    
        # returns 
        . dataFrame          : df, where each row represents files form one group, wiht one core name, 
                               eg: test_batch_01, test_batch_02 etc..., and rows names after  filetypes_dict keys
                               have corresponding filetypes, eg:
                               test_batch_01_features.npy, test_batch_01_labels.csv, 
                               additional columns shows also the path, and subset_type (key from subsets_dict)
               
        # Note
        the function find all files that mach any cobination of the following pattern
        > f'{subsets_dict[<key>]}*filetypes_dict[<key>]' 
                    
                    
    """
    
    os.chdir(path)
    filename_table_list = [] # one table with all file names,
    # ...
    for i, subset_name in enumerate(list(subsets_dict.keys())):
        " subset_file_name_pat may allow finidng one, or many different files wiht eg 01, 02, ... 0n numbers or other designations "
        " these shodul be "

        # ........................................................................................
        # get pattern used to find files from a given data subset/group
        subset_pat = subsets_dict[subset_name] # pat may be str or a list with >=1 str, 
        first_filetype = filetypes_dict[list(filetypes_dict.keys())[0]]

        
        # ........................................................................................
        # step 1. find first file, to later on final any othe file types in the same order
        one_subset_corename_with_one_filetype_list = []
        
        # if, there is a list, it means more then one pattern was mashing to a given subset, 
        if isinstance(subset_pat, list):
            for one_subset_pat in subset_pat:
                for file in glob.glob(f"{one_subset_pat}*{first_filetype}"):
                    one_subset_corename_with_one_filetype_list.append(file)            
        else:
            for file in glob.glob(f"{subset_pat}*{first_filetype}"):
                one_subset_corename_with_one_filetype_list.append(file)

            
        # ........................................................................................
        # step 2. find all different types of associated files defined by different file_affix_pat
        """ LIMITATION: these different types of files shdoul be in the same directory
        """
            
        # .. test if anything coult be found 
        if len(one_subset_corename_with_one_filetype_list)==0:
            if verbose==True:
                print(f"{subset_name} - No files were found using provided subset_pat_list & filetype_pat_list[0]")
            else:
                pass
            pass # becausde there is nothing to continue with, and I dont want to stop on that
            
        else:
            if verbose==True:
                print(f"{subset_name} - {len(one_subset_corename_with_one_filetype_list)} files were found, at least for the first filetype")            
            else:
                pass
                  
            # .. remove affix, and create core file names that can be used to find other types of files, 
            """ and create list of core files that can be used to search 
                for different types of files for each item in that list
            """
            one_subset_corename_list = pd.Series(one_subset_corename_with_one_filetype_list).str.split(first_filetype, expand=True).iloc[:, 0].values.tolist()

            
            # .. search filtypes for all core names, 
            for one_file_corename in one_subset_corename_list:
                
                # .... now find all filetypes with the same corename (one by one),
                one_corename_filetypenames_dict = dict()
                for filetype_name in list(filetypes_dict.keys()):

                    # - get patter used to filnd one filetype, 
                    filetype_pat = filetypes_dict[filetype_name]
                
                    # - search for ONE_FILE_NAME
                    ONE_FILE_NAME = [] # at least ot. shoudl be one !
                    for file in glob.glob(f"{one_file_corename}*{filetype_pat}"):
                        ONE_FILE_NAME.append(file) 

                    # - test if you can find only one name, if not the patterns provided are not specifficnc enought
                    if verbose==True:
                        if (ONE_FILE_NAME)==0:
                            print(f"Error - FILE NOT FOUND: {f'{one_file_corename}*{filetype_pat}'}")
                        if len(ONE_FILE_NAME)==1:
                            "everything is ok"
                            pass
                        if len(ONE_FILE_NAME)>1:
                            print(f"Error: provided combination of - {file_core_name} - and - {file_affix_pat}- is not speciffic enought !!!")
                            print("Error: in results more then one file was found and now only the first one will be loaded")
                    else:
                        pass

                    # .. add that file to the duct with assocuated files, 
                    one_corename_filetypenames_dict[filetype_name] = ONE_FILE_NAME[0]

     
                # .... finally, add each group of assicated files wiht the same core file name to filename_table 
                "ie. build table row"
                filename_table_list.append({
                   "subset_name": subset_name,
                   "path": path, 
                   **one_corename_filetypenames_dict
                })

    return pd.DataFrame(filename_table_list)
            
    
    
    
    
    
    
    
# Function, ......................................................................
# working version ...... 2020.12.11 ----- finally !!!!!!!!!!!!!!!
# Function, ......................................................................
def pair_files(*, search_patterns, pair_files_with, allow_duplicates_between_subsets=False, verbose=False, track_progres=False):
    '''
        function to find list speciffic files or pairs or groups of associated files, 
        eg: batch of images and their labels that can be with different formats and in different locations,
        One file type is described with so called corefilename and subset types that will allow to group them, 
        and search for other, associated files using that cofilename and profided filename prefixes and extensions, 
        
        done: 2020.12.10
        
        # inputs
        . search_patterns                  : dict, see example below
        . pair_files_with                  : a type of file that is parired with other filetypes
        . allow_duplicates_between_subsets : bool, if True, the function will stop on subset collection, 
                                           that assigned the same leading filenames to differetn subsets 
        . verbose                          : bool, 
        . track_progres                    : bool, like versbose, but sending minimal info on the process going on
        
        # returns:
        . dictionary with DataFames        : dict, key==Datasubsets collection
                                                   values=pd.DataFrame, with paired file_name's and file_path's
                                                   and col: subset_name that allows separating different subsets in one df
                                                   df, contains also several other values, that can help createing 
                                                   new derivative files
        
        
        # Example

            search_patterns = {
                "all_data":{    # one datasets collection will create one dataframe, 
                    "extracted_features":{
                        "file_path":PATH_extracted_features,
                        "file_prefix": f'{module_name}_{dataset_name}_{dataset_name}',
                        "file_extension": "_encoded.npy", 
                        "file_corename": {
                            "train": f"_",   # this will return several duplicates in train data
                            "valid": f"_valid_batch",
                            "test": f"_test_01",
                            "test_2": f"_test_02"  # you may add more then one in a list !
                        }},
                    "labels":{
                        "file_path":None,
                        "file_prefix": None,
                        "file_extension": "labels.csv"
                        },

                    },
                "partial_data":{    # one datasets collection will create one dataframe, 
                    "extracted_features":{
                        "file_path":PATH_extracted_features,
                        "file_prefix": f'{module_name}_{dataset_name}_{dataset_name}',
                        "file_extension": "_encoded.npy", 
                        "file_corename": {
                            "train": [f"_train_batch01", f"_train_batch02",f"_train_batch03",f"_train_batch03",f"_train_batch03"],
                            "valid": f"_valid_batch01",
                            "test": f"_test_01"
                        }},
                    "labels":{
                        "file_path":None,
                        "file_prefix": None,
                        "file_extension": "labels.csv"
                        },

                    }
            }  

            # .......
            df = pair_or_list_files(
                search_patterns=search_patterns, 
                pair_files_with="extracted_features", 
                verbose=True)
        
        
        
    '''    
    STOP_LOOP               = False # if true after some test, function stops execution and returns None
    subsets_collection_list = list(search_patterns.keys()) # used separately, and returned as dict with different tables,
    compare_all_files_to    = pair_files_with # legacy issue, I chnaged the name to make it more informative
    paired_filenames_dict   = dict() # keys==collection of subsets, values = table with paired filesnames/paths and name of the datasubset

    # -------------------------------------------------------------------------------   
    # create one df table per collection of subsets, 
    for subsets_collection_name in list(search_patterns.keys()):

        if track_progres==True:
            print("* Preparing: ", subsets_collection_name, " - from - ", subsets_collection_list)
        else:
            pass
        
        # -------------------------------------------------------------------------------   
        # Step 1. search filenames of the first filetype (compare_all_files_to !)
        # -------------------------------------------------------------------------------   
        '''
            here the df, is created with all items such as subsets_collection_name, & one_subset_name,
            that will allow identifying the file without the ptoblems, 
        '''
        
        # - list with subset names to loop over,
        subset_name_list_in_one_collection = list(search_patterns[subsets_collection_name][compare_all_files_to]["file_corename"].keys())
        
        # - list to store results on one subset collection (one entry == one file)
        one_subset_collection_file_list = list()
        
        # - loop over each subset
        for i, one_subset_name in enumerate(subset_name_list_in_one_collection):
            
            # **** STEP 1 **** parameters, , 
            
            # .... get variables provided as parameters to the function for one_subset_name, 
            file_path      = search_patterns[subsets_collection_name][compare_all_files_to]["file_path"] # str
            file_prefix    = search_patterns[subsets_collection_name][compare_all_files_to]["file_prefix"] # str
            file_extension = search_patterns[subsets_collection_name][compare_all_files_to]["file_extension"] # str
            file_corename  = search_patterns[subsets_collection_name][compare_all_files_to]["file_corename"][one_subset_name] # str/list

            # .... ensure that corename is a list, (can also be provided as str, with one pattern) 
            if isinstance(file_corename, str)==True:
                file_corename = [file_corename]
            else:
                pass
            
            
            # **** STEP 2 **** get filenames, 
            
            # .... set dir, 
            try:
                os.chdir(file_path)
            except:
                if verbose==True:
                    print(f"ERROR incorrect path provided for {compare_all_files_to}")
                else:
                    pass
                
            # .... identify all files in one subset from that subsets collection
            'all files found with all patterns added to the same list'
            found_file_name_list = []
            for one_file_corename in file_corename:
                for file in glob.glob(f"{file_prefix}*{one_file_corename}*{file_extension}"):
                    found_file_name_list.append(file)
                    
            # ... ensure there are no repeats in found_file_name_list 
            found_file_name_list = pd.Series(found_file_name_list).unique().tolist()
                    
                
                
            # **** STEP 3 **** get file speciffic corename and place all results in dict in the list                  
                
            # .... create a file_speciffic_corename
            file_speciffic_corename_s = pd.Series(found_file_name_list)
            file_speciffic_corename_s = file_speciffic_corename_s.str.replace(file_prefix, "")
            file_speciffic_corename_s = file_speciffic_corename_s.str.replace(file_extension, "")  

            # .... add each file into one_subset_collection_file_list
            for file_name, filespeciffic_corename in zip(found_file_name_list, file_speciffic_corename_s):
                one_subset_collection_file_list.append({
                    "subsets_collection_name": subsets_collection_name,
                    "subset_name": one_subset_name, 
                    f"{compare_all_files_to}_file_name": file_name, 
                    f"{compare_all_files_to}_file_path":file_path,
                    f"{compare_all_files_to}_file_prefix": file_prefix, 
                    f"{compare_all_files_to}_file_corename":file_corename,
                    f"{compare_all_files_to}_file_extension":file_extension,
                    f"{compare_all_files_to}_filespeciffic_corename":filespeciffic_corename,
                })
                
    
        # -------------------------------------------------------------------------------   
        # Step 2. test if all file_names are unique and were not used in mutiple subsets
        # -------------------------------------------------------------------------------
        'caution this maybe done intentionally' 
    
        # - get all filenames in a given cllection of subsets, - duplicates can be the same files listed for 2 different subsets, 
        collected_filenames = pd.DataFrame(one_subset_collection_file_list).loc[:, f"{compare_all_files_to}_file_name"]
        
        # - test it all are unique, 
        if collected_filenames.shape[0]!=len(collected_filenames.unique().tolist()):
            if allow_duplicates_between_subsets==False:
                STOP_LOOP = True
            else:
                STOP_LOOP = False # duplicates are not a problem :)
                pass
            
            if track_progres==True:
                print("ERROR, corename duplicates were detected in", subsets_collection_name, " -> function has been stoppped")
            else:
                pass            
            
            # .... print info with examples and placement of duplicates, 
            if verbose==True:
                # identify and print all filenames that are in more then one subset in one collection, 
                temp_df = pd.DataFrame(one_subset_collection_file_list)
                s = pd.Series(collected_filenames)
                # ...
                values_counted = s.value_counts()
                filenales_duplicated_in_different_subsets = pd.Series(list(values_counted.index))[(values_counted>1).values.tolist()].values.tolist()
                print("ERROR, following files were placed in more then one subset:")
                print(f"--- in --- Collection name (df): {subsets_collection_name}")
                for fi, one_f in enumerate(filenales_duplicated_in_different_subsets):
                    found_in = (temp_df.subset_name.loc[temp_df.loc[:, f"{compare_all_files_to}_file_name"]==one_f]).unique().tolist()
                    print(fi, one_f, ": ",found_in)
            else:
                pass
        else:
            if track_progres==True:
                print(" - corename duplicates were not detected in", subsets_collection_name)
            else:
                pass   
            
            STOP_LOOP = False # no duplicates detected, 
            pass
                
        # -----------------------------------------------------------------------------   
        # Step 3. Loop over all other filetypes, 
        #         and use filespeciffic corenames to find mashing files with each file 
        #         in one_subset_collection_file_list
        #-----------------------------------------------------------------------------  
        # before continuing
        if STOP_LOOP==True:
            'eg because we had no results at all, or unwanted duplicates between subsets'
            break
        else:
            
            # ...............................
            # OPTION A. RETURN RESULTS TO MAIN DICT
            # ...............................
            'finish if, that is the only collection of subsets' 
            if len(list(search_patterns[subsets_collection_name].keys()))==1:
                paired_filenames_dict[subsets_collection_name] = pd.DataFrame(one_subset_collection_file_list)
                
                
            # ...............................
            # OPTION B. OR FIND AT LEAST ONE MORE MATCHING FILE
            # ...............................                
            else:


                # -------------------------------------------------------------------------------   
                # Step 3A. Find what types of files shodul be paired 
                #.         wiht the typefile listed atfter step 1&2
                # -------------------------------------------------------------------------------                
                
                # - list all filetypes
                other_file_types = pd.Series(list(search_patterns[subsets_collection_name].keys()))
                
                # - excluude compare_all_files_to filetype that was already found in steps 1&2
                other_file_types = other_file_types.loc[other_file_types!=compare_all_files_to].values.tolist() 

                
                
                # -------------------------------------------------------------------------------   
                # Step 3B. search paired files from all other file types
                # -------------------------------------------------------------------------------        
                # Filetype, that is being added
                for one_file_type in other_file_types:
                    
                    # - info, 
                    if verbose==True:
                        print(f"\n{''.join(['.']*80)}\nPairing {compare_all_files_to} with {one_file_type}\n{''.join(['.']*80)}\n")
                    else:
                        pass 
                    
                    if track_progres==True:
                        print(f" - - pairing files with {one_file_type}")
                    else:
                        pass       
 


                    # -------------------------------------------------------------------------------   
                    # Step 3C. add new filetype to each row/dict 
                    #          in df/list creeated in stes 1&2
                    # -------------------------------------------------------------------------------      
                    # done one by one, 
                    for list_index_nr in range(len(one_subset_collection_file_list)):
                    
                    
                        # *** step 1 **** get all items to find matching filetype
                        
                        # --- collect base-file filespeciffic corename and subset_name
                        one_paired_group_files_dict    = one_subset_collection_file_list[list_index_nr]
                        #one_subset_name_for_one_file   = one_paired_group_files_dict["subset_name"] # not really required,
                        filespeciffic_corename         = one_paired_group_files_dict[f"{compare_all_files_to}_filespeciffic_corename"]

                        # --- collect new file path, prefix and extension
                        'caution if all are none, the softwae will select exctly the same file as corefile'
                        new_file_path = search_patterns[subsets_collection_name][one_file_type]["file_path"]
                        if new_file_path==None:
                            new_file_path = one_paired_group_files_dict[f"{compare_all_files_to}_file_path"]
                        else:
                            pass
                        new_file_prefix = search_patterns[subsets_collection_name][one_file_type]["file_prefix"]
                        
                        if new_file_prefix==None:
                            new_file_prefix = one_paired_group_files_dict[f"{compare_all_files_to}_file_prefix"]
                        else:
                            pass        
                        new_file_extension = search_patterns[subsets_collection_name][one_file_type]["file_extension"]
                        
                        if new_file_extension==None:
                            new_file_extension = one_paired_group_files_dict[f"{compare_all_files_to}_file_extension"]
                        else:
                            pass     
                  
                                                
                        # *** step 2 **** find new filetype/s mashing to requested patterns
                        try:
                            os.chdir(new_file_path)
                        except:
                            if verbose==True:
                                print(f"ERROR incorrect path provided for {one_file_type}")
                            else:
                                pass
                        # ...
                        identified_filenames_to_pair_list = []
                        for file in glob.glob(f"{new_file_prefix}*{filespeciffic_corename}*{new_file_extension}"):
                            identified_filenames_to_pair_list.append(file)

                            
         
                        # *** step 3 ****  test the results, and place the item back in the list, or report an error, 
                        
                        # --- correct results
                        if len(identified_filenames_to_pair_list)==1:
                            'correct result - my file has only one math with the other extension and/or prefix'
                            one_paired_group_files_dict[f"{one_file_type}_file_name"] = identified_filenames_to_pair_list[0]
                            one_paired_group_files_dict[f"{one_file_type}_file_path"] = new_file_path
                            if verbose==True:     
                                print(f'Pairing: {one_paired_group_files_dict[f"{compare_all_files_to}_file_name"]} + + + {identified_filenames_to_pair_list[0]}')
                            else:
                                pass
                        
                        # --- incorrect results
                        elif len(identified_filenames_to_pair_list)==0:
                            'incorrect result - no data'
                            one_paired_group_files_dict[f"{one_file_type}_file_name"] = np.nan
                            one_paired_group_files_dict[f"{one_file_type}_file_path"] = np.nan
                            if verbose==True:     
                                print(f'ERROR - NO MATCH for: {one_paired_group_files_dict[f"{compare_all_files_to}_file_name"]} + + + with {one_file_type}')
                            else:
                                pass    
                            
                        elif len(identified_filenames_to_pair_list)>1:
                            'incorrect result - my file has multiple mashtes with == filespeciffic corefilename was not speciffic enought'
                            one_paired_group_files_dict[f"{one_file_type}_file_name"] = np.nan
                            one_paired_group_files_dict[f"{one_file_type}_file_path"] = np.nan
                            if verbose==True:     
                                print(f'ERROR - MULTIPLE MATCHES for: {one_paired_group_files_dict[f"{compare_all_files_to}_file_name"]} + + + with {one_file_type}')
                                for ii, error_file_name in enumerate(identified_filenames_to_pair_list):
                                    print(f' -------- > {ii} {error_file_name}')
                            else:
                                pass     
                            
                            
                        # *** step 4 ****  return the results to collection list
                        one_subset_collection_file_list[list_index_nr] = one_paired_group_files_dict    
                      
                        
        # -------------------------------------------------------------------------------   
        # Step 4. place one_subset_collection_file_list ad pd df in the external dictionary
        # -------------------------------------------------------------------------------       
        paired_filenames_dict[subsets_collection_name] = pd.DataFrame(one_subset_collection_file_list)
                    
            
    # before continuing check if something has chnages in the status - this code prevents errors, from earlier on, 
    if STOP_LOOP==True:
        'eg because we had no results at all, or unwanted duplicates between subsets'
        return None
    else:        
        return  paired_filenames_dict
    
    



