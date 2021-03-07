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
import shutil
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import numpy as np
import pandas as pd
import warnings

# donwlonad required files from configs
from src.configs.project_configs import PROJECT_NAME
from src.configs.project_configs import DATASET_NAMES # datasset names with its description, 
from src.configs.project_configs import CLASS_COLORS # colors that I assigned to 7 classes in target variable
from src.configs.project_configs import CLASS_DESCRIPTION # information on each class, including fulll class name and diegnostic description
from src.configs.project_configs import TFHUB_MODELS # names of TF hub modules that I presenlected for featuress extraction with all relevant info,


# created derivate configs, :)
" this way you have to modify only configs in the main config file"


# derivative config, ........................................................
"""create to translate class_names in raw metadata into informative classnames used in the project
   add full_class_names to img_metadata
   prepare ditionary to map full names to class id in metadata table
"""
CLASSDX_TO_FULLCLASSNAME = dict(zip(
    list(CLASS_DESCRIPTION.keys()), 
    [CLASS_DESCRIPTION[x]["class_full_name"] for x in list(CLASS_DESCRIPTION.keys())]
))



# derivative config, ........................................................
"""class colors mapped onto informative classnames used in the project'
   example how it shoudl look like:
   CLASS_COLORS_FULLCLASSNAME={
      'Actinic_keratoses': 'steelblue',
     'Basal_cell_carcinoma': 'dimgrey',
      ....
      ....
      }
"""
CLASS_COLORS_FULLCLASSNAME = dict(zip(
    [CLASS_DESCRIPTION[x]["class_full_name"] for x in list(CLASS_DESCRIPTION.keys())], 
    [CLASS_COLORS[x] for x in list(CLASS_DESCRIPTION.keys())]
))






