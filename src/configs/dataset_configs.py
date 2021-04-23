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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# configs .......................................................
'''
    special value used in class labels, 
    to indicated class samples that shodul not be used with a given model
    This is used, only if all items in from a given class are removed, 
    otherwise, you must either use mask for a given dataset 
    or create new dataset without selected samples
'''
DROPOUT_VALUE = "to_dropout"


# configs .......................................................
'''
    colors assigned to original class labels, 
'''
CLASS_COLORS ={
 'bkl': 'orange',
 'nv': 'forestgreen',
 'df': 'purple',
 'mel': 'black',
 'vasc': 'red',
 'bcc': 'dimgrey',
 'akiec': 'steelblue'}



# configs .......................................................
'''
   to order which color I wish to display when merging classes together, 
   required to keep nice and menaingfull coloring of differetn classes 
   classes, wiht the highest numbers will be used as "color donor"
   - not required, but recommended if you like some colors or some of them have meaning, 
'''
CLASS_COLORS_zorder ={
 'bkl': 300,
 'nv':  500,
 'df': 1,
 'mel': 200,
 'vasc': 1,
 'bcc': 1,
 'akiec': 1}


# configs .......................................................
# CLASS_LABELS_CONFIGS
#   key                          : str, name of the classyficaiton system used 
#            "info"              : str, notes for the user
#            "class_labels_dict" : dict, key: original class label, value: labels used in that classyficaiton system 
#.   "melanoma_stat_labels_dict" : dict, custom dict, added to allow caulating accuracy statistucs, with one class containigni melanoma (POSITIVE),
#                                 vs all other classes designated as NEGATIVE
CLASS_LABELS_CONFIGS = {
  
  "Cancer_Detection_And_Classification":{   
      "info":"more informative class names for raw data",
      
      "class_labels_dict":{
         'akiec': 'Squamous_cell_carcinoma',
         'bcc': 'Basal_cell_carcinoma',
         'bkl': 'Benign_keratosis',
         'df': 'Dermatofibroma',
         'nv': 'Melanocytic_nevus',
         'mel': 'Melanoma',
         'vasc': 'Vascular_skin_lesions'},
    
      "melanoma_stat_labels_dict":{
         'Squamous_cell_carcinoma': 'NEGATIVE',
         'Basal_cell_carcinoma': 'NEGATIVE',
         'Benign_keratosis': 'NEGATIVE',
         'Dermatofibroma': 'NEGATIVE',
         'Melanocytic_nevus': 'NEGATIVE',
         'Vascular_skin_lesions':'NEGATIVE',
         'Melanoma': 'POSITIVE'}
    },
  
  "Cancer_Risk_Groups":{   
      "info":"""
               7 original classes were grouped into three oncological risk groups  
               with vasc&nv assigned into low lever skin lessions, all other cancer types into cancer benign, 
               and melanoma as separate category
              """,
      
      "class_labels_dict":{
         'akiec': 'Medium-benign_cancer',
         'bcc': 'Medium-benign_cancer',
         'bkl': 'Medium-benign_cancer',
         'df': 'Medium-benign_cancer',
         'nv': 'Low-skin_lession',
         'mel': 'High-melanoma',
         'vasc': 'Low-skin_lession'},
      
      "melanoma_stat_labels_dict":{
         'Low-skin_lession': 'NEGATIVE',
         'Medium-benign_cancer': 'NEGATIVE',
         'High-melanoma': 'POSITIVE'}
    }
  }


# configs .......................................................
'''
    list of labels used for each dataset, 
    each dataset has a unique composition of files/images
    these files/images can be assign to different classes
    you can not add more images, to a given dataset (in that case you shodul create a new one)
    but you can remove images, from the dataset during training or prediciton steps, using DROPUOT value, insteade of class label

'''
DATASET_CONFIGS = {  
 "HAM10000": {
     "info": "raw data grouped with original classes, no augmentation, no duplicate removal",
     "labels": [
                "Cancer_Detection_And_Classification", 
                "Cancer_Risk_Groups"
               ]
   },
   "HAM10000_workshop": {
     "info": "small dataset prepared to demonstrate PyClass funcitons and pipeline for creating summary in SkinDiagnosticAI feasibiility study",
     "labels": [
                "Cancer_Detection_And_Classification", 
                "Cancer_Risk_Groups"
               ]
   }
}



# config function, ..............................................
def add_dataset_class_colors_to_CLASS_LABELS_CONFIGS(CLASS_COLORS, CLASS_LABELS_CONFIGS, CLASS_COLORS_zorder=None):
    '''
        function will add class_colors to each datasets, using CLASS_COLORS from
        original dataset class names, This way, there is only one place where 
        the colors must be changed,
        ..
        CLASS_COLORS, DATASET_CONFIGS, CLASS_COLORS_zorder : all shodul be in this config file, 
    '''

    # find colors of each class,
    'if some classes were merged, the color of the last one will be used'
    for dataset_name in list(CLASS_LABELS_CONFIGS):
        class_labels_dict = CLASS_LABELS_CONFIGS[dataset_name]["class_labels_dict"]
        
        # find colors of each class,
        if CLASS_COLORS_zorder==None:
            'if some classes were merged, the color of the last one will be used'
            CLASS_LABELS_CONFIGS[dataset_name]["class_labels_colors"] = dict(
                    zip([dataset_class_names[x] for x in list(class_labels_dict.keys())],
                        [CLASS_COLORS[x] for x in list(class_labels_dict.keys())]))

        else:
            'I use zorder to determine which color shodul be used in merged classes'  
            temp_df = pd.DataFrame([
                CLASS_COLORS,
                CLASS_COLORS_zorder,
                class_labels_dict
            ], index=["CLASS_COLORS", "CLASS_COLORS_zorder", "class_labels_dict"]).transpose()
            temp_df

            class_labels_colors = dict()
            for one_class_name in temp_df.class_labels_dict.unique().tolist():
                dfs = temp_df.loc[temp_df.class_labels_dict==one_class_name,:]
                class_labels_colors[one_class_name] = dfs.sort_values("CLASS_COLORS_zorder", ascending=False).CLASS_COLORS.iloc[0]

            CLASS_LABELS_CONFIGS[dataset_name]["class_labels_colors"] = class_labels_colors
    
    return  CLASS_LABELS_CONFIGS


# update configs ......................................................................................
CLASS_LABELS_CONFIGS = add_dataset_class_colors_to_CLASS_LABELS_CONFIGS(
    CLASS_COLORS, 
    CLASS_LABELS_CONFIGS, 
    CLASS_COLORS_zorder
)




# config function, ......................................................................................
def add_class_encoding_decoding_to_CLASS_LABELS_CONFIGS(CLASS_LABELS_CONFIGS):
    '''
        helper function, required for smooth running of ML models
        adds following dictionaries to CLASS_LABELS_CONFIGS:
        . class_encoding - class_label  -> unique digit
        . class_decoding - unique digit -> class_label
    '''
    # find colors of each class,
    'if some classes were merged, the color of the last one will be used'
    for dataset_variant in list(CLASS_LABELS_CONFIGS.keys()):
        
        # prepare class encoding/decoding
        class_labels_dict     = CLASS_LABELS_CONFIGS[dataset_variant]["class_labels_dict"]
        class_names_to_encode = pd.Series([class_labels_dict[k] for k in list(class_labels_dict.keys())]).unique().tolist()
        # .....
        class_encoding        = dict(zip(class_names_to_encode, list(range(len(class_names_to_encode)))))
        # .....
        class_decoding = dict()
        for k in list(class_encoding.keys()):
            class_decoding[class_encoding[k]]=k
            
        # place class encoding/decoding in class_labels_conf...           
        CLASS_LABELS_CONFIGS[dataset_variant]["class_encoding"] = class_encoding
        CLASS_LABELS_CONFIGS[dataset_variant]["class_decoding"] = class_decoding
        
    return CLASS_LABELS_CONFIGS
        
    
# update configs ......................................................................................
CLASS_LABELS_CONFIGS = add_class_encoding_decoding_to_CLASS_LABELS_CONFIGS(CLASS_LABELS_CONFIGS)  


