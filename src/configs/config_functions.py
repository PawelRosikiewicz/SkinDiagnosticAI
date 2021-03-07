# ********************************************************************************** #
#                                                                                    #
#   Project: SkinAnaliticAI                                                          #                                                         
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz_gmail.com                                                  #
#                                                                                    #
#.  This notebook is a part of Skin AanaliticAI development kit, created             #
#.  for evaluation of public datasets used for skin cancer detection with            #
#.  large number of AI models and data preparation pipelines.                        #
#                                                                                    #     
#   License: MIT                                                                     #
#.  Copyright (C) 2021.01.30 Pawel Rosikiewicz                                       #
#   https://opensource.org/licenses/MIT                                              # 
#                                                                                    #
# ********************************************************************************** #

'''
    these are special functions create only for that project, 
    these are not runned automatically, but with variables defined in config files in each notebook, 
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from src.utils.new_data_loaders import pair_files


# Config fucntion, ..............................................................................
def DEFINE_DATASETS(*, 
    dataset_name,
    dataset_variants,
    module_names,
    path,
    path_labels=None,
    verbose=False,
):
    '''
        config function to create collections of datasubsets 
        each collections is a dataframe with at least 5 columns
        dictionary provided withiung that function is custom build for a given project, 
        for more info on dict examples see: help to pair_files() from src.utils.new_data_loaders
    '''
    
    # empty object
    results_list = list()
    
    # you may work with many dataset varinats, but only one dataset, 
    for dv_i, dataset_variant in enumerate(dataset_variants): 
        # set path for labels,                                      
        path_extracted_features = os.path.join(path, f"{dataset_name}__{dataset_variant}__extracted_features")
        if path_labels==None:
            used_path_labels=path_extracted_features     # use other name, because it will be None only for the first iteration - yep i wasted 1h of my work                                         
        else:
            used_path_labels=path_labels
        
        # info
        if verbose==True: 
            print(f"\n{''.join(['*']*70)}", f"\n- {dv_i} - {dataset_name}, {dataset_variant}\n", f"{''.join(['*']*70)}")
        else:
            pass
              
            
        # ** / MANUAL PART 1 / ******************************************************
        '''
          here you define what are the subset colections, eg all_data
          and define which of them are going to be used as train, valid and test (keys in dict)
          values are the names of dataset subsets used in MANUAL PART 2
          important:
            you need to define at least train dataset
            . valid and test are optional, and if empty, introduce value None, 
            . test value accepts list with several test subsets
            . valid subset, acceopts also a float, and if used, this will 
              be a fraction of train dataset taken to validation dataset
          
        '''
        DATA_SUBSET_ROLES={
            "small_subset_data":{
                "train":"train",
                "valid": "valid",
                "test":["test", "test_2"]
            },
            "all_data":{
                "train":"train",
                "valid": "valid",
                "test":["test", "test_2"]
            }
          }
            

  
        # ** / MANUAL PART 2 / ******************************************************            
        '''
            Second manual part, where you must define paths to files with extracted features 
            from images and corresponding batch labels, It was created becaues you may introduce 
            many different batch labels, as long as batches of data are organized in the same way, 
            you may also introduce DROPOUT value o some files/images to exclude them from analysis 
            all these chnages are created by adding different dataset_varinats, meanin you chnage the 
        '''       
        # folders may contain features extracted with many different modules, 
        for m_i, module_name in enumerate(module_names):
            if verbose==True: 
                print(f"\n .... {m_i} .... {dataset_variant}, {module_name}")
            else:
                pass            
                                                             
            # prepare variables for the function that finds pairs of matrix with extracted features + file labels, 
            search_patterns = {
                        "all_data":{    # one datasets collection will create one dataframe, 
                            "extracted_features":{
                                "file_path":path_extracted_features,
                                "file_prefix": f'{module_name}', # it can be much more complicated, and have dataset name and dataet varinat included, or some version name etc...
                                "file_extension": f"encoded.npy", 
                                "file_corename": {
                                    "train":  "train",   # this will return several duplicates in train data
                                    "valid":  "valid",
                                    "test":   "test_01",
                                    "test_2": "test_02"  # you may add more then one in a list !
                                }},
                            "labels":{
                                "file_path":used_path_labels, # ie, these files are inside the same folder, 
                                "file_prefix": None, # same extesion as in the extracted_features
                                "file_extension": "labels.csv"
                                },

                            },
                        "small_subset_data":{    # one datasets collection will create one dataframe, 
                            "extracted_features":{
                                "file_path":path_extracted_features,
                                "file_prefix": f'{module_name}',
                                "file_extension": f"encoded.npy", 
                                "file_corename": {
                                    "train": ["train_01", "train_02"],
                                    "valid":  "valid_01",
                                    "test":   "test_01",
                                    "test_2": "test_02"  # you may add more then one in a list !
                                }},
                            "labels":{
                                "file_path":used_path_labels,
                                "file_prefix": None,
                                "file_extension": "labels.csv"
                                },

                            }
                    }  

            # find paired fails, for each dataset_name + dataset_variant combination
            paired_files_df = pair_files(
                search_patterns=search_patterns,
                pair_files_with="extracted_features",
                track_progres=verbose,
                verbose=False
            )
            
            # to to results_dict
            for k in paired_files_df.keys():
                results_list.append({
                    "dataset_name": dataset_name, 
                    "dataset_variant": dataset_variant,
                    "module_name":module_name,
                    "subset_collection_name": k,
                    "subset_composition_list":paired_files_df[k].subset_name.tolist(),
                    "dataset_compositions_df": paired_files_df[k] # the tables that has all the data,   
                })
            
    return results_list, DATA_SUBSET_ROLES
         