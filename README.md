![skindiagnosticai title image](images/skindiagnosticai_title_image.png)

# SkinDiagnosticAI; 
## Detecting Skin Cancer with Low cost devices and AI technology
## Project: Feasibility study with open source data and PyClass AI workbench
      
Author: __Pawel Rosikiewicz, Founder, and Team Leader at SwissAI__  
Contact: prosikiewicz@gmail.com    
License: __MIT__    
ttps://opensource.org/licenses/MIT        
Copyright (C) 2021.01.30 Pawel Rosikiewicz  

## __About SkinDiagnosticAI__
Skin  Diagnostic AI is an open source project created to develop application for medical specialists, and patients that will help identifying potential skin cancer with low cost devices, and collect dermatoscopic images for medical community. It will also help to patients to store their images in easy and accessible way on personal devices, allow remote communication with the doctor, provide preliminary diagnosis, such as cancer rrisk assesment. Furthermore, SkinDiagnosticAI application will allow, collecting, clustering and storing mutlitple images from the same patient, or time, thus allowing for faster, and more speciffic patient-doctor interaction in a limited time-span. 

More info on SkinDiagnosticAI project: https://simpleai.ch/skin-diagnostic-ai/

## __About the project__
### __Feasibility study__   
Notebooks presented in this repository are part of feasibility study conducted for SkinDiagnosticAI initiative.   
The study was conducted in five major steps, with the following goals: 
* __Step 1.__ To create PyClass - an automated piepline for analysis and classysfication of medical images
  * the piepline was developed using Swissroad dataset, that was smalled and easier to analize for non-medical specialist then dermatoscopic images,
  * you may fid it here: https://github.com/PawelRosikiewicz/Swissroads
* __Step 2.__ To compare large number of AI models, with different diagnostic purposes using open source data from __Harvard database__ with skin images diagnosed by the group of medical experts
  * speciffic goals were as follow:
    - to identify main challenges with anylsis and classyfication of dermatosciopic images   
    - to explore different strategies for data preparation, treatment and feature extraction,
    - to test, of the shelf AI solutions, with extensive grid search, 
    - to develope baseline for further analyses,  and model development  
    - to evaluate what statistics and error fucntions should be used for developing final, and ensemble models,
    - to compare different methods for results presentation that are most usefull for medical experts and non-medical users
  > NOTE: selected models and PyClass outputs can be used as Proof Of Concept, or an early stage MVP, thanks to reporting capability of the PyClass
* __Step 3.__ to evaluate __business value proposition__ of different models, and consult them with users and domain experts
* __Step 4.__ __to deploy the pipeline__ with selected models on the cloud, and use it as __Proof of Concept__ produc, that can be used to generate actionable results, 
  > NOTE: `Steps. ` to 4 were conducted iteratively, several times`
* __Step 5.__ to perform __AI readiness assesment__, and to collect a set of final requirements for potential MVP
  > NOTE: preliminary buisness proposal, ai feasibility assement, and  was create before stage 1. 
  > Here we used used the knowledge collected in feasibility study to update opur estimates,

### __What is in this GitHub repository?__
The notebooks, and software presented in this repository were used to conduct Steps 1-3. 
The `slides`attached below the text show selected results from all steps 1-5

## __About PyClass__
>  main tool used to conduct feasibility study

PyClass is an open-sourse, AI workbench for development of classyficiton models for medical images.   
Main functions are as follow:
* data cleaing, and preparation for keras image dgenerators, and feature extraction with pre-trained convolution networks, 
* EDA, on medical images, that are typicall difficult to disstiguos for non-medical specialist (see examples on slides below)
* automated comparison and selection of large number of models, TF feature extreaction networks, and image classyfication schemas.   
* model optimization,   
* error analysis     
* baseline selection. 
* summary anylsis on different granulaty of paremter spaces, 

> Note: PyClass, was developed using only basic, python libraries, such as scipy, tensofrflow, and matplolib, and can be used with any version of python >3.6
> See how PyClass was used for develoment of reliable classification models of vehicles on roads https://github.com/PawelRosikiewicz/Swissroads






## __Data Preparation with PyClass for SkinDiagnosticAI__

> current example can be run as is, all configs and data were already prepared for You
> If you would like to create `your own project wiht PyClass`, or add new models and conditions to that project, you need need to start with project_suetup steps 1-3, that were described in separate notebooks, see `project_setup/01 - 03 ... notebooks`

* __01. Set PyClass enviroment__
    * NOTEBOOK: __`notebooks/01_Setup.ipynb`__ 
    * in this step you will:
      * Setting Up Project Enviroment with proper directory structure
      * clone/download PyClass repository, 
      * Download the data, and tf-hub models for feature extraction (optional)
      * Prepare config files for the project
        * `tfhub_configs.py`: configs on tf hub modules used for feature extraction
        * `project_configs.py`: basic description of the dataset
        * `dataset_configs.py`: contains dictionaries used to label images in each class, provide colors etc, select classes for cutom statistics, etc,,
        * `config_functions.py`: special functions used to select files for data processing and module training,
    > NOTE: each config file contains detailed intrsuctions, on how to create them, 

* __02. Data Preparation__
    * Data preparation, using PyClass functions was done in`project_setup/02_a-c... notebooks`
    * it includes, donwloading the images fron Harvard dataverse, or Kaggle, (links are in notebooks)
    * cleaning, the images, and organizing in train/test/valid folders, with separate folders containg images from each class, named after class name
    * dividing large dataset into smaller subsets, or 
    * __Caution__ data used for de-novo cnn models, are not divided into smaller subsets, these are kept in one folder to allow keras image generators creating augmented images, while trainign the model,
 
 
   
* __03. Feature extraction__
    * PyClass provides you with two options, You may either use `ne-novo cnn networks` for classyficaiton of images, or you may use `keras and skilearn models`, on extracted features with pretrained convolutional networks, downloaded from tf-hub, or other location.
    * to extract features follwo the instructions in `project_setup/02_a-c... notebook`
    * models: 
        * option 1. you may use url, that needs to be added to `model_configs`
        * option 2. you may donwload the model to `model/` directory, and use them with the same function
   


## __EDA with PyClass for SkinDiagnosticA__
> PyClass provides automated functions for EDA, with the data prepared for model training

* it allows answering the following questions:
     * __1. To answer following questions__
         * what is the composition of the dataset?
         * does different classes have similar size? 
         * can we find hidden subclasses, and if yes, are they well represented, 
         * are there some obvious problems with the dataset
         * do I have sufficient number of images for making image classyfication
         * why some classes are imbalanced (if they are)            
                            
     * __2. To plot image examples from each class__
        * image examples are provided with several other functions, or separately, with the function `plot_n_image_examples`, as below
                     
     * __3. Technical Exploration__  
        * to compare results of feature extraction using
                * hierarchical clustering with examples, 
                * PCA
        * to understand what preprocessing steps, can be required to bild the most optimal baseline, tranfer learning and deep learnign models,   



## __Introduction to PyClass with SkindDiagnosticAI, in interactive enviroment__ 

### __about the workshop__
I prepared a short introduction to PyClass, and SkinDiagnosticAI feasibility study for Applied Mashine Leanring Days (AMLD 2021),
In this course, you may use PyClass a subset of HAM10000 dataset, ie. the same dataset that I used with SkinDiagnosticAI feasibility study
The code can be run, and modified in vitrual enviroment, thanks on Renku platform provided by __Swiss Data Science Center, SDSC__

### __to play with the code, follow these instructions:__
* Got to: SkinDiagnosticAI, with PyClass implemented at SDCS  
  https://renkulab.io/projects/swissai/amld-2021-workshop
* Click on `Environments`
* Start new interactive environment (Click on `NEW`)
* in new window, set following parameters:
  * Number of CPUs: 2
  * Amount of Memory: 2G
  * leave branch, commit and Def. Environment as they are,
* Click on `Start Environment` button (below all parameters)
  * you may wait for preparation of the environment
* Click `Connect` (blue button on the left)
* Now, you should be connected to virtual environment with Jupiter lab open
* Notebooks 1-4 are in “notebook” folder
  * you may open and try each of them, 
  * all functions, and coffings are in “src” directory 
  * the files were annotated, with instructions how to modify them, 
  * The code was explained on my workshop @AMLD: 
* follow the instreuctions withint each notebook, 
  * all functions have help available ... help(<function_name>)
  



## Presentation on SkinDiagnosticAI Project
* the slides shows full analyis done on over 5000 compared models and data treatment procedures, 
* Jupyter notebooks in notebook folder shows light vervion of that analyis that can be reapeated by the user and build up to any number of compared models, 
* for more information see: My presentation on SkinDiagnosticAI project: https://youtu.be/W624gdkDqRQ?t=491

> all images were created wiht PyClass AI workbech

![skindiagnosticai presentation slide](images/Slide1.png)
![skindiagnosticai presentation slide](images/Slide2.png)
![skindiagnosticai presentation slide](images/Slide3.png)
![skindiagnosticai presentation slide](images/Slide4.png)
![skindiagnosticai presentation slide](images/Slide5.png)
![skindiagnosticai presentation slide](images/Slide6.png)
![skindiagnosticai presentation slide](images/Slide7.png)
![skindiagnosticai presentation slide](images/Slide8.png)
![skindiagnosticai presentation slide](images/Slide9.png)
![skindiagnosticai presentation slide](images/Slide10.png)
![skindiagnosticai presentation slide](images/Slide11.png)
![skindiagnosticai presentation slide](images/Slide12.png)
![skindiagnosticai presentation slide](images/Slide13.png)
![skindiagnosticai presentation slide](images/Slide14.png)
![skindiagnosticai presentation slide](images/Slide15.png)
![skindiagnosticai presentation slide](images/Slide16.png)
![skindiagnosticai presentation slide](images/Slide17.png)
![skindiagnosticai presentation slide](images/Slide18.png)
![skindiagnosticai presentation slide](images/Slide19.png)
