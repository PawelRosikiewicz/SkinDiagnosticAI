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
import shutil
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell pattern
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Function, ................................................................................................
def copy_and_organize_files_for_keras_image_generators(
    # ... files descirption
    file_name_list,           # list, names of files to be copied, 
    class_name_list,          # list of classses, same lenght as file_names_list, Caution, no special characters allowed !
    # ... inputs
    src_path,                 # str, path to file, that holds at least one, specified folder with files (eg images.jpg) to copy, to  data_dst_path/class_name/files eg .jpg
    src_dataset_name_list,    # names of directories, that shodul be found in input_data_path
    # ... outputs
    dst_path,                 # str, path to file, where the new_dataset file will be created, as follow data_dst_path/dataset_name/subset_name/class_name/files eg .jpg
    dst_dataset_name,         # str, used to save the file, in data_dst_path/dataset_name/subset_name/class_name/files eg .jpg
    dst_subset_name ="train", # str, same as above, eg=train data, data_dst_path/dataset_name/subset_name/class_name/files eg .jpg
    file_extension_list=[""], # file extensions that shoudl be tranferred, dont forget about the dot., the fucntion will also accept "" as for no extension
    # ...
    verbose=False,
    track_progres=False,  
    return_logfiles=False,     # returns, list with 
    create_only_logfiles=False # bool, special option, the function bechaves in the same way, except it do not copy files, but create logfiles only,
                               # with classified, items, grouped in dct where key is a class name, it rtunrs two logfiles, with present and mising files,
):
    
    """
        A function that takes any number of folders located in one directory, input_data_path, 
        list of file names (image names) and associated list of classes of each of these files, 
    
         Caution, if you wish to copy files with diffferent extension but the same name, 
                  run that function separately for all extension, or add them into file_name_list
                  for as long as the files have different names, nothing will be overwritten or removed,
                  
                  
        # inputs
        . file_name_list,           : list, names of files to be copied, 
        . class_name_list,          : list of classses, same lenght as file_names_list, 
                                    Caution, no special characters allowed !
        # ... input data
        . src_path,                 : str, path to file, that holds at least one, specified folder with files 
                                    (eg images.jpg) to copy, to  data_dst_path/class_name/files eg .jpg
        . src_dataset_name_list,    : list, names of directories, that shodul be found in input_data_path
        
        # ... output path 
        . dst_path,                 : str, path to file, where the new_dataset file will be created, 
                                         as follow data_dst_path/dataset_name/subset_name/class_name/files eg .jpg
        . dst_dataset_name,         : str, used to save the file, 
                                         in data_dst_path/dataset_name/subset_name/class_name/files eg .jpg
        . dst_subset_name ="train", : str, same as above, eg=train data, 
                                         data_dst_path/dataset_name/subset_name/class_name/files eg .jpg
        . file_extension_list=[""], : file extensions that shoudl be tranferred, dont forget about the dot., 
                                         the fucntion will also accept "" as for no extension
        # ...
        . verbose                   : bool, 
        . track_progres             : bool, if True, periodically displays mesage on progres in the proces  
        . return_logfiles           : bool, returns, list with 
        . create_only_logfiles      : bool, special option, the function bechaves in the same way, 
                                          except it do not copy files, but create logfiles only,
                                          with classified, items, grouped in dct where key is a class name, 
                                          it rtunrs two logfiles, with present and mising files,                                         
        # Returns
        . copy of each file specified in file_name_list in data_dst_path/dataset_name/subset_name/class_name/files eg .jpg
        . copied_file_names_dict   : dict, key = class name, values, in a list, = fcopied file names
        . missing_file_names_dict  : dict, key = class name, values, in a list, = file names that were 
                                     not found in searched directories,    
                  
    """

    # test if img_nages_list and img_class_names_list, have the same lenght
    if verbose==True:
        if len(file_name_list)==len(class_name_list):
            print(f"Preparing {len(file_name_list)} to organize into file structure required by keras image data generators\n")
        else:
            print("ERROR! img_names_list, and img_class_names_list have different lenght")
    else:
        pass

    
    # ................................................
    # create directory to store an entire dataset,
    try:
        os.mkdir(os.path.join(dst_path, dst_dataset_name))
        os.chdir(os.path.join(dst_path, dst_dataset_name)) # test if everything is correct
        if track_progres==True or verbose==True:
          print(f"DATASET created file: {os.path.join(dst_path, dst_dataset_name)} created\n")
        else:
          pass
    except: 
        if track_progres==True or verbose==True:
          print(f"DATASET file already exist (or was not created): {os.path.join(dst_path, dst_dataset_name)}")
        else:
          pass
        
    
    # no create directory for a speciffic data subset created by that function, 
    try:
        os.chdir(os.path.join(dst_path, dst_dataset_name))
        os.mkdir(os.path.join(dst_path, dst_dataset_name, dst_subset_name))
        if track_progres==True or verbose==True:
          print(f"DATA_SUBSET file created file: {os.path.join(dst_path, dst_dataset_name, dst_subset_name)}\n")
        else:
          pass
    except: 
        if track_progres==True or verbose==True:
          print(f"DATA_SUBSET file already exist (or was not created): {os.path.join(dst_path, dst_dataset_name, dst_subset_name)}")
        else:
          pass
        

        
    # ................................................
    # Create logfiles for copied and missing files, requessted by file_name_list

    # create dict that will store list of tranferred images, and images that counlt be find, but were present in metadata table
    copied_file_names_dict = dict()  # key == classname
    missing_file_names_dict = dict()  # key == classname

    # .. add list for each class that will be created,
    for one_class_name in pd.Series(class_name_list).unique().tolist():
        copied_file_names_dict[one_class_name] = list()  # key == classname
        missing_file_names_dict[one_class_name] = list() # key == classname        

    # ................................................
    # Copy all available images, listed in metadata table, into new directory with train dataset

    # .. copy all images, to proper directories, if directories are not found, create them,
    class_number = 0 # for verbose messages 
    for i, (one_file_name, one_class_name) in enumerate(zip(file_name_list, class_name_list)):

        # .... info, 
        if track_progres==True:
            if i%100==0 and i>0:
                print(f"...{i}_images.copied", end="")
            else:
                pass
        else:
            pass

        # .... create new directory, if not present already !!!!
        os.chdir(os.path.join(dst_path, dst_dataset_name, dst_subset_name))
        try:
            dst_one_class_path = os.path.join(dst_path, dst_dataset_name, dst_subset_name, one_class_name)
            os.mkdir(dst_one_class_path)
            if track_progres==True or verbose==True:
              print(f"\nCLASS_FILE was created {class_number}: {dst_one_class_path}\n")
            else:
              pass
            class_number +=1
        except: 
            pass # file already exist, or PATH_results was not created correctly  
            # dont write any message or else it will be displayed many times
        
        # .... searcvh for the image, and when found, COPY/PASTE it to new directory,
        '''caution, I pooled images from some directories, on EDA,
           thus sometimess I had duplicates, the function will use 
           only the first image that it founds in the list of directories
        '''
        for src_one_dataset_potential_path in [os.path.join(src_path, x) for x in src_dataset_name_list]:

            # find if any file has requested name and extension in the file, 
            # --- if yes, stop looking for more
            file_names_found_list = []
            os.chdir(src_one_dataset_potential_path)
            # this loop allows to search for the same file name with different file extensions
            for one_file_extension in file_extension_list:
                
                # .. here you may add more files, if there are duplicates located  - they shoudlnd be possible, but, !
                for file in glob.glob(f"{one_file_name}{one_file_extension}"): # find any img oin any format of that name,  
                    file_names_found_list.append(file)     

            # check if you must ssearch for more, or you found whatever you were looking for, 
            if len(file_names_found_list)>0:
                'No, I found what I was looking for - smooth criminal, nice music :)'
                # os.chdir(one_img_potential_input_direcory)
                src_one_dataset_confirmed_path = src_one_dataset_potential_path
                coppied_one_file_name_with_extension = file_names_found_list[0]   # only first one should be coppied !
                break
            if len(logfiles)==0:
                src_one_dataset_confirmed_path = None
                coppied_one_file_name_with_extension = None    # only first one should be coppied !
                
    
        # .... finally, tranfer the file to new location from confirmed/first found, old location
        if src_one_dataset_confirmed_path==None:
             missing_file_names_dict[one_class_name].append(one_file_name)
        else:
            if create_only_logfiles==False:
                shutil.copy(
                    os.path.join(src_one_dataset_confirmed_path, coppied_one_file_name_with_extension),
                    os.path.join(dst_one_class_path, coppied_one_file_name_with_extension)
                )
            else:
                pass
            copied_file_names_dict[one_class_name].append(coppied_one_file_name_with_extension)

    # report on potential missing data/files that were not found 
    if verbose==True:
        print("\n\nSUMMARY:...............................................\n")
        for cl_i, one_class_name in enumerate(list(copied_file_names_dict.keys())):
            print(f"\n{cl_i}-{one_class_name}: files  copied: {len(copied_file_names_dict[one_class_name])}")
            print(f"{cl_i}-{one_class_name}: files missing: {len(missing_file_names_dict[one_class_name])}")
    else:
        pass
            
    # and return logfiles, 
    if create_only_logfiles==True or return_logfiles==True:
        return copied_file_names_dict, missing_file_names_dict
    else:
        pass
      
      
      
      
      
      
      
      
      
# Function, ...............................................................................
def create_file_catalogue(*, path, searched_class_name_list = None, verbose=False):
    '''
        create a dict. with class name and all files inside folder of that class name, 
        path     : str, full path to ffolder that holds class.-folders with eg images
        searched_class_name_list : list, with strings, classnames -folders wiht files to catalogue that you wish to specify, 
                  eg in case only a certain classes shodul be searched, 
        verbose  : bool,  
    '''
    # ... logfile, 
    file_catalogue = dict()

    # go to directory, and visit every single file, 
    os.chdir(path)
    if searched_class_name_list==None:
        items_in_path_list = os.listdir(path)
    else:
        items_in_path_list = searched_class_name_list 

    # get names of all files in that directory
    for item in items_in_path_list:
        try:
            os.chdir(os.path.join(path, item)) # to be sure it is a directory, 
            file_catalogue[item] = os.listdir(os.path.join(path, item))
        except:
            pass

    # summary
    if verbose==True:
        for item in list(file_catalogue.keys()):
            print(f'{item}: {len(file_catalogue[item])}')
    else:
        pass
        
    return file_catalogue



  
  
  
# Function, ................................................................................
def create_keras_comptatible_file_subset_with_class_folders(*,
    file_names_dict,
    move_files=False, 
    random_state_nr=None,
    # ...
    src_path,
    src_subset_name,    
    dst_path,
    dst_subset_name,
    # ... 
    subset_size=0.1, 
    min_nr_of_files_per_class_to_copy=None,
    max_nr_of_files_per_class_to_copy=None,
    fixed_subset_size=None, # int, or None, fixed number of files in. each class will be moved/copied
    # ...
    verbose=False                                                        
):
    """
        This function can be used to create test or any other subset of files, 
        - as input it uses catalogued files in directory that is organized as follow src/src_subset/class_name/{files to tranfer}
        - it will recreated the same structure dst/dst_subset/class_name/{files to tranfer} in a new location
        - src and dst may be the same, 
        - additionally, it creates a subset with specified number of files that are either moved or copied, (see params)
        
        # Inputs:
        . file_catalogue_dict      : output from create_file_catalogue()
                                     whre key=class name, values in the list == file names with extension !            
        . move_files               : bool,  if False, files are copied not moved, 
        . random_state_nr          : int, or None, 
          ...
        . src_path                 : str, full path to directory that contains source dataset 
                                     see: src_path/src_subset/class_name/{files to tranfer}
        . src_subset_name          : str, eg. "train"  
                                     file name that contains folders of class names with files to move/copy,
                                     see: src_path/src_subset/class_name/{files to tranfer}                        
          ...
        . dst_path                 = as src_path, can be the same as src_path
        . dst_subset_name          = as src_subset_name, eg "test"
          ... 
        . subset_size              : float (0, 1), or int >0,
                                     if float, it is used as proportion to tranfer proportion of requested files, 
                                     is int, the given number of items will be tranferred, 
        . min_nr_of_files_per_class_to_copy: int, or None, 
        . max_nr_of_files_per_class_to_copy: int, or None, 
        . fixed_subset_size        :# int, or None, fixed number of files in. each class will be moved/copied
                                    Caution, it will overwrite, min/max and int form of subset_size
                                    it will allow copy/moveing only max nr of files available in a given class
          ...
        . verbose                  : bool, 
    
        # Returns:
        . dict                     : with pd.dataFrame for each class in src_path/src_subset/class_name/{files to tranfer}
                                     and columns with names of all moved/copied files, and full source and destination paths
                                     used to tranfer these files, and info whther they were moved or copied only
    """

    # IMPORTANT MESSAGE
    if verbose==True:
        if move_files==True:
            print(f" CAUTION -")
            print(f" move_files is set True, all selecxted files will be moved, not copied")
            print(f" in case you made the mistake, you may use logfiles returned by this function to reverse that")
            print(f" the logfile is a pd.DataFrame with full path of source and destination for eacvh tranfewrred file\n")
        else:
            pass
    else:
        pass


    # * declare path to full path to directory wht a subset that shoudl be tranferred and created, 
    src_subset_path = os.path.join(src_path, src_subset_name)
    dst_subset_path = os.path.join(dst_path, dst_subset_name)

    if src_subset_path==dst_subset_path:
        if verbose==True:
            print("ERROR path to source and destination are the same - THE FUNCTION WAS NOT EXECUTED !!!!!")
            print("please use at least different subset name, eg test, when createing a subset from eg train data")
        else:
            pass
    else:
        ' you may continue :)'

        # * create directory for new subset of data, 
        try:
            os.chdir(dst_path)
            os.mkdir(dst_subset_path)
            if verbose==True:
                print(f"Following direcotry was created: {dst_subset_path}")
            else:
                pass
            # to test if new directory was crrated properly, 
            try:
                os.chdir(dst_subset_path)
            except:
                if verbose==True:
                    print("ERROR, dir. for new data subset could not be accessed, or created, make sure you dont use special character in it")
                else:
                    pass       
        except:
            if verbose==True:
                print("ERROR: destination path may be incorrect")
            else:
                pass

            
        # * tranfer files from each directory listed in file_catalogue_dict
        logfile = dict()
        for one_class_name in list(file_names_dict.keys()):


            # Step 0. Create directory for new class in a subset directory, 

            # ... path to one class in src/dst paths
            src_subset_one_class_path = os.path.join(src_subset_path, one_class_name)
            dst_subset_one_class_path = os.path.join(dst_subset_path, one_class_name)

            # ... create new foled for files fromthat one class, 
            try:
                os.chdir(dst_subset_path)
                os.mkdir(dst_subset_one_class_path)
                os.chdir(dst_subset_one_class_path)
            except:
                try:
                    os.chdir(dst_subset_one_class_path)
                    if verbose==True:
                        print(f"Caution: files in class - {one_class_name} - will be copied to pre-existing directory, some files may be overwrittend")
                    else:
                        pass
                except:
                    if verbose==True:
                        print(f"ERROR: following directory was not created or can not be acccessed {dst_subset_one_class_path}")
                    else:
                        pass


            # Step 1. Decide how many images to copy/remove, in a goven class

            # .... create requested subset size
            if subset_size<1:
                nr_of_files_to_tranfer = np.ceil(len(file_names_dict[one_class_name]))*subset_size
            if subset_size>1:
                nr_of_files_to_tranfer = int(subset_size)
            else:
                pass

            # .... chech if poposed nr, of files is not larger/smaller then min,max, and nr of files in the given file, 
            if min_nr_of_files_per_class_to_copy!=None:
                if nr_of_files_to_tranfer<min_nr_of_files_per_class_to_copy:
                    nr_of_files_to_tranfer = int(min_nr_of_files_per_class_to_copy) 
                else:
                    pass
            else:
                pass

            if max_nr_of_files_per_class_to_copy!=None:
                if nr_of_files_to_tranfer>max_nr_of_files_per_class_to_copy:
                    nr_of_files_to_tranfer = int(max_nr_of_files_per_class_to_copy)                                                     
                else:
                    pass
            else:
                pass

            # .... OVERWRITE THE ABOVE number if fixed nr of images was requested, 
            if fixed_subset_size!=None:
                nr_of_files_to_tranfer = int(fixed_subset_size)
            else:
                pass

            # .... but, check if you have enought files in the class 
            if nr_of_files_to_tranfer>len(file_names_dict[one_class_name]):
                nr_of_files_to_tranfer = len(file_names_dict[one_class_name]) # ie. tranfer all available files, in that class
                if verbose==True:
                    print(f"CAUTION: {one_class_name} has {len(file_names_dict[one_class_name])} that all will be moved/copied")
                else:
                    pass
            else:
                pass              

            # ............ info .............
            if verbose==True:
                print("Class -",one_class_name ,"- has -", len(file_names_dict[one_class_name]), 
                      "- files and -", int(nr_of_files_to_tranfer), "- will be moved/copied")
            else:
                pass


            # Step 2. Create list with files naes in that class to move/copy
            if int(nr_of_files_to_tranfer)<len(file_names_dict[one_class_name]):
                if random_state_nr!=None:
                    np.random.seed(random_state_nr)
                else:
                    pass
                selected_filess_idx = np.random.choice(a=np.arange(len(file_names_dict[one_class_name])), size=int(nr_of_files_to_tranfer), replace=False)
                selected_file_names_list = pd.Series(file_names_dict[one_class_name]).iloc[selected_filess_idx].values.tolist()
                
            elif int(nr_of_files_to_tranfer)==len(file_names_dict[one_class_name]):
                 selected_file_names_list = file_names_dict[one_class_name]
            
            elif int(nr_of_files_to_tranfer)==0:
                 selected_file_names_list = None          
            

            # Step 3. move/copy files one by one, 
            logfile[one_class_name] = []
            if selected_file_names_list!=None:
                for i, one_file_name in enumerate(selected_file_names_list):
                    src = os.path.join(src_subset_one_class_path, one_file_name)
                    dst = os.path.join(dst_subset_one_class_path, one_file_name)

                    if move_files==False:
                        'only copy the file'
                        shutil.copy( src, dst)
                    else:
                        'the file will be moved from dst ro src location'
                        shutil.move( src, dst)            

                    # .. add this file to logfiles
                    logfile[one_class_name].append(
                         {"one_file_name": one_file_name,
                         "src": src,
                         "dst": dst,
                         "move_files":move_files
                        })
            else:
                # .. add this file to logfiles
                logfile[one_class_name].append(
                         {"one_file_name": None,
                         "src": None,
                         "dst": None,
                         "move_files":move_files
                        })                


            # Step 4. turn log file dict for one class into pd.DataFrame
            logfile[one_class_name] = pd.DataFrame(logfile[one_class_name])

        return logfile


      
      
      
      
      
      
      
      
      
      
      
# Function, .....................................................................
def create_data_subsets(*,
    src_path,
    src_subset_name,     
    dst_path=None,
    dst_subset_name_list=None,
    new_subset_size=0.1,
    min_new_subset_size=0.02,                     
    move_files=False,
    random_state_nr=0,
    fix_random_nr = False, # works only if you use MOVE file==True, to ensure reproducibility, 
    verbose=False   
):
    
    """
        Nice function to create data subsets, form datasets ordered as follow
        src_dataset/class_name_folder/{files in that class}
        IMPORTANT, if move_files==True, the files are movesd from src to dst in order of appearance in the lists wiht dst file names, 
        ....
        when more files is requested to move, copy then it is available in source filder, 
        == then the last or the only batch is eaither smaller or slightly bigger, 
        
        # Inputs:
        # ........................................................................................
        . move_files               : bool,  if False, files are copied not moved, 
        . random_state_nr          : int,  any number, 
        . fix_random_nr            : bool, if True, used only with move_files ==True, 
                                     to use the same exact nardo nr for each subset taken form original folder, with file moving
                                     thus ensiring that the same division of files can be repeated form the same starting conditions
          ...
        . src_path                 : str, full path to directory that contains source dataset 
                                     see: src_path/src_subset/class_name/{files to tranfer}
        . src_subset_name          : str, eg. "train"  
                                     file name that contains folders of class names with files to move/copy,
                                     see: src_path/src_subset/class_name/{files to tranfer}                        
        . dst_path                 : same as src_path, can be the same as src_path
          ...
        . dst_subset_name_list     :list, with str, each same as as src_subset_name, eg "test"
        . new_subset_size          : float (0, 1), IT MUST BE >0 and <1 
                                     proportion to tranfer proportion of requested files, 
                                     if single values used, it will be used to all files in dst_subset_name_list 
        . min_new_subset_size      : no new subset will be smaller then this proportion, 
                                     if the last batch woudl be smaller then this value, it will be added to the one before, 
          ...
        . verbose                  : bool, 

        # Returns:
        # ........................................................................................
        . dict                     : src_dataset/dst_subset_name/{files in that class}
                                     for each pair of items in dst_subset_name_list & new_subset_size
    """   
    
    
    
    
    # set up names & directories, 
    if dst_path==None:
        dst_path = src_path
    else:
        pass
    
    # ..
    if dst_subset_name_list==None:
        dst_subset_name_list = [f"{src_subset_name}_subset"]
    else:
        pass   
    
    
    # set random numbers for files selection
    if fix_random_nr==True and move_files==True:
        random_numbers = [random_state_nr]*len(dst_subset_name_list)
    else:
        'made to avoid having the same set of samples in a subset at any cost !'
        'it can be a problme in case you wish to build amazingly large number os subset'
        np.random.seed(random_state_nr)
        random_numbers = np.unique(np.random.randint(0, 10000000, len(dst_subset_name_list)*100))

        
    # if all subset are of the same size, 
    if isinstance(new_subset_size, list): 
        new_subset_proportion_list = new_subset_size
    else: 
        new_subset_proportion_list = [new_subset_size]*len(dst_subset_name_list)

        
        
        
    # create subsets, 
    for i, (dst_subset_name, new_subset_proportion, randint)  in enumerate(zip(dst_subset_name_list, new_subset_proportion_list, random_numbers)): 

        # test if the batch size is not smaller then the one se as min, size
        if new_subset_proportion <min_new_subset_size:
            new_subset_proportion=min_new_subset_size
        else:
            pass
        
        # calculate adjusted % of files taken at each iteration,
        'if the files are moved, the % of taken values must be rising with each iteration'
        if i == 0 or move_files==False:
            "either start from 1, or keep it that way, if you copy files"
            total_pop = 1 
        else:
            pass
        adjusted_new_subset_proportion = new_subset_proportion/total_pop
        total_pop -= (total_pop*adjusted_new_subset_proportion)

        
        
        # decide what to do when end-size subsets appears, 
        # .. use Image generator to find out how many files there are, and display small message
        if verbose==True:              
            print(f"\n{''.join(['.']*80)}\n- {i} - PREPARING: {dst_subset_name}\n{''.join(['.']*80)}\n")
            if move_files==False:   print(f"{new_subset_proportion} of files with be COPIED\n")
            if move_files==True:    print(f"{new_subset_proportion} of files with be MOVED\n")
            print(f"from: {src_subset_name}")
            countgen = ImageDataGenerator()
            iterator = countgen.flow_from_directory(os.path.join(src_path, src_subset_name))          
        else:
            'just find how many files is left in the subset'
            countgen = ImageDataGenerator()
            iterator = countgen.flow_from_directory(os.path.join(src_path, src_subset_name)) 
        

        # STOP CONDITIONS or assocatied parameters for the last batch that can be smaller or larger then the rest batches, 
        
        # ... stop in case new_subset_size>=1
        if new_subset_proportion>=1:
            if verbose==True:
                print("CAUTION, any new_subset_size should be always smaller then 1")
                " there was an error in original function that do not allow me to use value >=1, but it can be 0.99999999"
            else:
                pass
            break
        else:
            pass
        
        # ... stop if there is nothing to copy or to transfer, 
        if len(iterator.filenames)==0:
            if verbose==True:
                print("NO FILES IN SOURCE DIRECTORS THAT CAN BE COPPIED or MOVED --- NO ACTIONS WERE TAKEN, check your files !")
                break
        else:
            "but if passed continue, and check if there is no need to chnage batch size for the last dataset"
            if (1-total_pop)<=min_new_subset_size or total_pop<0 or adjusted_new_subset_proportion>=1:
                # .. take all to the  files ion the remaning dataset
                adjusted_new_subset_proportion = 0.99999999999 # this is to deal with some small error in my script, but that condition will be ignored wiht set size below
                create_subset_kwargs = {
                    "min_nr_of_files_per_class_to_copy" : 1,
                    "max_nr_of_files_per_class_to_copy" : None,
                    "fixed_subset_size" : len(iterator.filenames)       
                }

            else:
                create_subset_kwargs = {
                    "min_nr_of_files_per_class_to_copy" : 1,
                    "max_nr_of_files_per_class_to_copy" : None,
                    "fixed_subset_size" : None       
                }


        # First catalogue all files in source dataset
        files_catalogue = create_file_catalogue(
            path = os.path.join(src_path, src_subset_name),
            searched_class_name_list = None, # if none, catalog, all, 
            verbose=False)

        ## ...
        logs = create_keras_comptatible_file_subset_with_class_folders(
            file_names_dict          = files_catalogue, # output from create_file_catalogue()
            move_files               = move_files, # if False, files are copied not moved, 
            src_path                 = src_path,
            src_subset_name          = src_subset_name,      
            dst_path                 = dst_path,
            dst_subset_name          = dst_subset_name,
            subset_size              = adjusted_new_subset_proportion,
            random_state_nr          = randint,
            **create_subset_kwargs
        )

        if verbose==True:
            print(f"\nto: {dst_subset_name}")
            countgen = ImageDataGenerator()
            iterator = countgen.flow_from_directory(os.path.join(dst_path, dst_subset_name))
            if move_files==False: 
                print("prop_left: 1.0", "; prop_taken:", new_subset_proportion, "; adjusted_prop_taken: ", adjusted_new_subset_proportion)
            else: 
                if (1-total_pop)<=min_new_subset_size or total_pop<0:
                    print("THIS IS THE LAST BATCH ---- all remaining files  in src dir were transfered or coppied")
                else:
                    print("prop_left: ",total_pop, "; prop_taken:", new_subset_proportion, "; adjusted_prop_taken: ", adjusted_new_subset_proportion)                 
        else:
            pass
        
        # break the loop if there is nothign left to do, 
        if (1-total_pop)<=min_new_subset_size or total_pop<0:
            break
        else:
            pass
        
        