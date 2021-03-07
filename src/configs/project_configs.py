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
PROJECT_NAME = "Skin_cancer_detection_and_classyfication"

# config, ...........................................................................................
# CLASS_DESCRIPTION
#. "key"                   :  str, class name used in original dataset downloaded form databse 
#      "original_name"     :  str, same as the key, but you can introduce other values in case its necessarly
#      "class_full_name"   :  str, class name used on images, saved data etc, (more descriptive then class names, or sometimes the same according to situation)
#      "class_group"       :  str, group of classes, if the classes are hierarchical, 
#      "class_description" :  str, used as notes, or for class description available for the user/client
#      "links"             :  list,  with link to more data, on each class, 
CLASS_DESCRIPTION = {
  'akiec':{
    "original_name":'akiec',
    "class_full_name": "squamous_cell_carcinoma", # prevoisly called "Actinic_keratoses" in my dataset, but ths name is easier to find in online resourses, noth names are correct,  
    "class_group": "Tumour_Benign",
    "class_description": "Class that contains two subclasses:(A) Actinic_Keratoses or (B) Bowen’s disease. Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma (Bowen’s disease) are common non-invasive, variants of squamous cell carcinoma that can be treated locally without surgery. These lesions may progress to invasive squamous cell carcinoma – which is usually not pigmented. Both neoplasms commonly show surface scaling and commonly are devoid of pigment, Actinic keratoses are more common on the face and Bowen’s disease is more common on other body sites. Because both types are induced by UV-light the surrounding skin is usually typified by severe sun damaged except in cases of Bowen’s disease that are caused by human papilloma virus infection and not by UV. Pigmented variants exist for Bowen’s disease and for actinic keratoses",
    "links":["https://dermoscopedia.org/Actinic_keratosis_/_Bowen%27s_disease_/_keratoacanthoma_/_squamous_cell_carcinoma"]
    },
  
  'bcc':{
    "original_name":'bcc',
    "class_full_name": "Basal_cell_carcinoma",
    "class_group": "Tumour_Benign",
    "class_description": "Basal cell carcinoma (BCC) is the most common type of skin cancer in the world that rarely metastasizes but grows destructively if untreated. It appears in different morphologic variants (flat, nodular, pigmented, cystic). There are multiple histopathologic subtypes of BCC including superficial, nodular, morpheaform/sclerosing/infiltrative, fibroepithelioma of Pinkus, microcytic adnexal and baso-squamous cell BCC. Each subtype can be clinically pigmented or non-pigmented. It is not uncommon for BCCs to display pigment on dermoscopy with up to 30% of clinically non-pigmented BCCs revealing pigment on dermoscopy. Based on the degree of pigmentation, some BCCs can mimic melanomas or other pigmented skin lesions. Depending on the subtype of BCC and the degree of pigmentation, the clinical differential diagnosis can be quite broad ranging from benign inflammatory lesions to melanoma. Fortunately, the dermoscopic criteria for BCC are visible irrespective of the size of the tumor and can be well distiguished using dermatoscopy",
    "links":["https://dermoscopedia.org/Basal_cell_carcinoma"]
    }, 
  
  'bkl':{
    "original_name":'bkl',
    "class_full_name": "Benign_keratosis", 
    "class_group": "Tumour_Benign",
    "class_description":  "Benign keratosis is a generic group that includes three typesy of non-carcinogenig lesions: (A) seborrheic keratoses (senile wart), (B) solar lentigo - which can be regarded a flat variant of seborrheic keratosis, (C) and lichen-planus like keratoses (LPLK), which corresponds to a seborrheic keratosis or a solar lentigo with inflammation and regression. The three subgroups may look different dermatoscopically, but we grouped them together because they are similar biologically and often reported under the same generic term histopathologically. Briefly: Seborrheic keratoses (A) are benign epithelial lesions that can appear on any part of the body except for the mucous membranes, palms, and soles. The lesions are quite prevalent in people older than 30 years.  Early seborrheic keratoses are light - to dark brown oval macules with sharply demarcated borders. As the lesions progress, they transform into plaques with a waxy or stuck-on appearance, often with follicular plugs scattered over their surfaces. The size of the lesions varies from a few millimeters to a few centimeters. Solar lentigines (B) are sharply circumscribed, uniformly pigmented macules that are located predominantly on the sun-exposed areas of the skin, such as the dorsum of the hands, the shoulders, and the scalp. Lentigines are a result of hyperplasia of keratinocytes and melanocytes, with increased accumulation of melanin in the keratinocytes. They are induced by ultraviolet light exposure. Unlike freckles, solar lentigines persist indefinitely. Nearly 90% of Caucasians over the age of 60 years have these lesions. LPLK (C), is one of the common benign neoplasms of the skin, and it is highly variable in its appearance, Some LPKL can show morphologic features mimicking melanoma and are often biopsied or excised for diagnostic reasons",
    "links": ["https://dermoscopedia.org/Solar_lentigines_/_seborrheic_keratoses_/_lichen_planus-like_keratosis"]
  },
  
  'df': {
    "original_name":'df',
    "class_full_name": "Dermatofibroma", 
    "class_group": "Tumour_Benign",
    "class_description": "Dermatofibromas (DFs) are prevalent cutaneous lesions that most frequently affect young to middle-aged adults, with a slight predominance in females. Clinically, dermatofibromas appear as firm, single or multiple papules/nodules with a relatively smooth surface and predilection for the lower extremities. Characteristically, upon lateral compression of the skin surrounding dermatofibromas, the tumors tend to pucker inward producing a dimple-like depression in the overlying skin; a feature known as the dimple or Fitzpatrick’s sign. Dermatofibroma is a benign skin lesion regarded as either a benign proliferation or an inflammatory reaction to minimal trauma. The most common dermatoscopic presentation is reticular lines at the periphery with a central white patch denoting fibrosis",
    "links": ["https://dermoscopedia.org/Dermatofibromas"]
  },
  
  'nv': {    
    "original_name":'nv',
    "class_full_name": "Melanocytic_nevus", 
    "class_group": "Tumour_Benign",
    "class_description": "Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants, which all were included in train data used for diagnosis. The variants may differ significantly from a dermatoscopic point of view. Unlike, melanoma they are usually symmetric with regard to the distribution of color and structure",
    "links":["https://dermoscopedia.org/Benign_Melanocytic_lesions"]
  }, 
  
  "mel": {
     "original_name":'mel',
      "class_full_name": "Melanoma", 
      "class_group": "Tumour_Malignant",
      "class_description": "Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants. If excised in an early stage it can be cured by simple surgical excision. Melanomas can be invasive or non-invasive (in situ). Melanomas are usually, albeit not always, chaotic, and some melanoma specific criteria depend on anatomic site, All variants of melanoma including melanoma in situ, except for non-pigmented, subungual, ocular or mucosal melanoma were included in train dataset used for diagnosis",
      "linkss": ["https://dermoscopedia.org/Melanoma"]
  }, 
  
  'vasc':{
      "original_name":'vasc',
      "class_full_name": "Vascular_skin_lesions", 
      "class_group": "Vascular_skin_lesions",
      "class_description": "Angiomas are dermatoscopically characterized by red or purple color and solid, well circumscribed structures known as red clods or lacunes.Data Used for training for diagnosis: Vascular skin lesions in the dataset range from cherry angiomas to angiokeratomas and pyogenic granulomas. Hemorrhage is also included in this category",
      "links": ["https://dermoscopedia.org/Vascular_lesions"]
  }
}