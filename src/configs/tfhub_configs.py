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


# config, ...........................................................................................
# Purpose: create config file for tf-hub module used, 
# Localization: tfhub_configs.py
# values:
#         "module_name"    : str, name used on plots, and for file saving
#         "working_name"   : str, alternative to module_name (eg shorter, version wihtout special characters)
#         "file_name"      : str, the name of the file donwloaded from tfhub, wiht a given module, (can be custom)
#         "module_url"     : str, url, to the module on tfhub      
#         "input_size"     : tuple,  (height, width) in pixes 
#         "output_size"    : int,  lenght of vector with extracted features, 
#         "note"           : str, notes, whatether you this is important, for other users

TFHUB_MODELS = {
    "MobileNet_v2": {
        "module_name": "MobileNet_v2",
        "working_name": "mobilenet",
        "file_name": "imagenet_mobilenet_v2_100_224_feature_vector_2", 
        "module_url": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2", 
        "input_size":(224, 224),
        "output_size": 1280,
        "note": ""
        },  
    "BiT_M_Resnet101":{
        "module_name": "BiT_M_Resnet101", 
        "working_name": "resnet",
        "file_name":  "bit_m-r101x1_1",
        "module_url":"https://tfhub.dev/google/bit/m-r101x1/1",      
        "input_size":  (224, 224),
        "output_size": 2048,
        "note":"tested on swissroads dataset, where it worked very well"
        }   
}# end