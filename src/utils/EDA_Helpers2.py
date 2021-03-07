# ************************************************************************* #
#                                                                           #
#                                                                           #
#                                                                           #  
#              - - -           EDA Helpers            - - -                 #     
#                      part of  DataFrame Explorer                          #
#                                                                           #
#            A package that I created to load, explore and summarize        #
#                      large dataFrames with examples                       #
#                                                                           #
#                       last update: 2021.02.26                             #
#                                                                           #
#   Author: Pawel Rosikiewicz                                               #
#   Contact: prosikiewicz_gmail.com                                         #
#                                                                           # 
#   License: MIT                                                            #
#   Copyright (C) 2021.01.30 Pawel Rosikiewicz                              #
#   https://opensource.org/licenses/MIT                                     #
#                                                                           #
#                                                                           #
# ************************************************************************* #






# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import random
import glob
import re
import os
import seaborn as sns



# Function, ............................................................................

def find_and_display_patter_in_series(*, series, pattern):
    "I used that function when i don't remeber full name of a given column"
    res = series.loc[series.str.contains(pattern)]
    return res





# Function, ............................................................................
 
def summarize_df(*, df, nr_of_examples_per_category = 3, csv_file_name="none", save_dir="none", verbose=True):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Summary table, with basic information on column in large dataframes,
                            can be applied to dafarames of all sizes, Used to create summary plots
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * df                DataFrame to summarize
        * nr_of_examples_per_category
                            how many, top/most frequent records shoudl 
                            be collected in each column and used as examples of data inputs form that column
                            NaN, are ignored, unless column has only NaN
        
        . Saving .          The fgunction return Dataframe, even if file name and path to save_dir are not available
                            In that case the file are not saved.
        * csv_file_name     .csv file name that will be used to save all three dataFrames create with that function
        * save_dir          path
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * data_examples.    DataFrame, summary of df, with the follwing values for each column imn df
                            . name	                 : column name in df, attribute name
                            . dtype.                 : {"nan", if ony NaN werer detected in df, "object", "numeric"}
                            . NaN_perc               : percentage of missing data (np.nan, pd.null) in df in a given attirbute
                            . summary                : shor informtaion on type and number of data we shoudl expectc:
                                                       if dtype == "numeric": return min, mean and max values
                                                       if dtype == "object" : return number of unique classes
                                                              or messsage "all nonnull values are unique"
                                                       if dtype == nan       : return "Missing data Only"                        
                            . examples                : str, with reqwuested number of most frequent value examples in a given category
                            . nr_of_unique_values	  : numeric, scount of all unique values in a category 
                            . nr_of_non_null_values   : numeric, count of all non-null values in a category
        
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()

        * top_val_names     DataFrame, same as top_val_perc, but with values, saved as string
 
    """    
    # info
    if csv_file_name!="none" and save_dir!="none":
        if verbose == True:
            print("\n! CAUTION ! csv_file_name shoudl be provided wihtout .csv file extension!")
    
    
    # imports to check if you have numeric or string type columns
    from pandas.api.types import is_string_dtype
    from pandas.api.types import is_numeric_dtype
    from pandas.api.types import is_datetime64_any_dtype

    # create df,
    col_names   =["All_values_are_unique", "Nr_of_unique_values", "Nr_of_non_null_values", 
                  "Examples","dtype", "nr_of_all_rows_in_original_df"]
    df_examples = pd.DataFrame(np.zeros([df.shape[1],len(col_names)]), columns=col_names, dtype="object")

    # add category names
    df_examples["name"] = df.columns

    # add NaN percentage,
    nan_counts_per_category = df.isnull().sum(axis=0)
    my_data = pd.DataFrame(np.round((nan_counts_per_category/df.shape[0])*100, 5), dtype="float64")
    df_examples["NaN_perc"] = my_data.reset_index(drop=True)

    # add nr of no NaN values
    my_data = df.shape[0]-nan_counts_per_category
    df_examples["Nr_of_non_null_values"] = my_data.reset_index(drop=True)

    # add "nr_of_all_rows_in_original_df"
    df_examples["nr_of_all_rows_in_original_df"] = df.shape[0]
    
    # these arr will be filled for future bar plot
    arr_example_percentage = np.zeros([df_examples.shape[0],nr_of_examples_per_category])
    arr_example_values     = np.zeros([df_examples.shape[0],nr_of_examples_per_category],dtype="object")
    
    # add examples and counts
    for i, j in enumerate(list(df.columns)):

        # add general data  ..............................................

        # number of unique nonnull values in each column,
        df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = df.loc[:,j].dropna().unique().size

        #  internal function helpers .....................................

        # categorical data, fillin_All_values_are_unique
        def fillin_All_values_are_unique(*, df_examples, df):
            if (df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]==0).values[0]: 
                return "Missing data Only"
            if ((df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]>0).values[0]) and (df.loc[:,j].dropna().unique().size==df.loc[:,j].dropna().shape[0]): 
                return "all nonnull values are unique"
            if ((df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]>0).values[0]) and (df.loc[:,j].dropna().unique().size!=df.loc[:,j].dropna().shape[0]): 
                return f"{int(df_examples.Nr_of_unique_values[df_examples.name==j].values[0])} classes"

        # fill other columns ..............................................

        # this is auto-fill in case there is no data
        if df[j].isnull().sum()==df.shape[0]:
            # (df_examples.loc[df_examples.name==j,"NaN_perc"]==100).values[0]: this value was rounded up/down and was returning false positives!!!!
            df_examples.loc[df_examples.name==j,"All_values_are_unique"] = "missing data only"
            df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"] = 0 # it should be 0, but i overwrite it just in case
            df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = 0 # it should be 0, but i overwrite it just in case
            df_examples.loc[df_examples.name==j,"Examples"] = "missing data only"
            df_examples.loc[df_examples.name==j,"dtype"] = "missing data only" # because I dont want to use that in any further reading

        # in other cases, we can create data examples, from nonnull values, depending on their type,
        else:

            if is_string_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "text"
                
                # All_values_are_unique,
                # use helper function, to find if there are only unique categorical values eg: url, place example in dct,  
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = fillin_All_values_are_unique(df_examples=df_examples, df=df)

                # Examples,
                count_noNa_values_sorted = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                perc_noNa_values_sorted  = count_noNa_values_sorted/np.sum(count_noNa_values_sorted)*100
                s        = perc_noNa_values_sorted[0:nr_of_examples_per_category].round(1)
                ind      = pd.Series(s.index).str[0:12].values.tolist()
                df_examples.loc[df_examples.name==j,"Examples"] = ";".join([str((x,y)) for x,y in zip(["".join([str(x),"%"]) for x in list(s)],ind)])
   
                # add examples to arr for plot
                arr_example_percentage[df_examples.name==j,0:s.values.size]=s.values
                arr_example_values[df_examples.name==j,0:s.values.size]= np.array(s.index)


            if is_numeric_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "numeric"
                
                # All_values_are_unique,
                x = list(df.loc[:,j].dropna().describe()[["min", "mean", "max"]].round(3))
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = f"{round(x[0],2)} // {round(x[1],2)} // {round(x[2],2)}"

                # Examples,
                ordered_values = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                ordered_values = (ordered_values/df.loc[:,j].dropna().shape[0])*100
                df_examples.loc[df_examples.name==j,"Examples"] = ";".join([str((x,y)) for x,y in zip(
                    ["".join([str(int(np.ceil(x))),"%"]) for x in list(ordered_values)][0:nr_of_examples_per_category],
                    list(ordered_values.index.values.round(3))[0:nr_of_examples_per_category])])

                # add examples to arr for plot
                vn = np.array(ordered_values.index)[0:nr_of_examples_per_category]
                vp = ordered_values.values[0:nr_of_examples_per_category]
                arr_example_values[df_examples.name==j,0:ordered_values.size] = vn
                arr_example_percentage[df_examples.name==j,0:ordered_values.size] = vp
                
            if is_datetime64_any_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "datetime"
                
                # variable summary,
                first_and_last_date = [str(x) for x in list(df.loc[:,j].dropna().describe()[["first", "last"]].dt.strftime('%b %d %Y'))]
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = f"{first_and_last_date[0]} - {first_and_last_date[1]}"

                # Examples,
                ordered_values = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                ordered_values = (ordered_values/df.loc[:,j].dropna().shape[0])*100
                df_examples.loc[df_examples.name==j,"Examples"] =";".join([str((x,y)) for x,y in zip(
                    ["".join([str(np.round(x,3)),"%"]) for x in list(ordered_values)][0:2], 
                    list(pd.Series(ordered_values.index[0:2]).dt.strftime('%b-%d-%Y %H:%M').values))])# i add only two, because these are really long, 

                # add examples to arr for plot
                vn = list(ordered_values.index)[0:nr_of_examples_per_category]
                vp = ordered_values.values[0:nr_of_examples_per_category]
                arr_example_values[df_examples.name==j,0:ordered_values.size] = vn
                arr_example_percentage[df_examples.name==j,0:ordered_values.size] = vp     
                
    # reorder column, so the look nicer, when displayed, 
    df_examples = df_examples.loc[:,["name", "dtype", 'NaN_perc', 'All_values_are_unique', 'Examples', 
                                     'Nr_of_unique_values', 'Nr_of_non_null_values', 'nr_of_all_rows_in_original_df']]   
    # rename some collumns, for PEP8 compliance,
    df_examples = df_examples.rename(columns={'All_values_are_unique':'summary', "Examples": "examples", 
                                              "Nr_of_unique_values": "nr_of_unique_values", "Nr_of_non_null_values":"nr_of_non_null_values"})
    

    # turn two additional elements into df's, 
    df_example_values = pd.DataFrame(arr_example_values, index=df_examples.name)
    df_example_percentage = pd.DataFrame(arr_example_percentage, index=df_examples.name)
    
    #### save the files 
    if csv_file_name!="none" and save_dir!="none":
        try:
            os.chdir(save_dir)
            df_examples.to_csv("".join([csv_file_name,".csv"]), encoding='utf-8', index=False)
            df_example_values.to_csv("".join(["top_val_names_",csv_file_name,".csv"]), encoding='utf-8', index=False)
            df_example_percentage.to_csv("".join(["top_val_perc_",csv_file_name,".csv"]), encoding='utf-8', index=False)
            
            # info to display,and table example, 
            if verbose==True:
                print(f"""{"".join(["."]*40)} \n the file: {csv_file_name} was correctly saved \n in: {os.getcwd()} \n{"".join(["."]*40)}""") 
                print(f"MY NEW TABLE WITH EXAMPLES \n")
                from IPython.display import display
                display(df_examples.head(2))

        except:
            if verbose==True:
                Error_message = "THE FILE WAS NOT SAVED, \n save_dir and/or csv_file_name were incorrect, or one of them was not provided"
                print(f"""{"".join(["."]*40)},\n ERROR,\n the file: {csv_file_name},\n {Error_message} \n in: {os.getcwd()} \n{"".join(["."]*40)}""")

    else:
        if verbose==True:
            Error_message = "THE FILE WAS NOT SAVED, \n save_dir and/or csv_file_name were not provided "
            print(f"""{"".join(["."]*40)},\n CAUTION,\n the file: {csv_file_name},\n {Error_message} \n in: {os.getcwd()} \n{"".join(["."]*40)}""")
    
    return df_examples, df_example_values, df_example_percentage







# Function, ............................................................................
 
def pie_chart_with_perc_of_classes_df_column(*, ax, s, title, font_size=10):    
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Pie chart to diplay categorical data, with % and numerical values in ticks
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        * ax                fig axis from matplotlib.pyplot
        * s                 Pandas, series with repeated records that will be counted and displayes as pie chart pieces
                            ! caution ! Mosre then 5 classes may cause problmes, in that case its better to to use
                            barplot.
        .
        * title             str, ax.set_title("title")
        * font_size         int, ticks fontsize
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * matplotlib 
          figure axis object
        
        * example           https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html
    """
        
    # create description for each calls with its percentage in df column
    s = s.value_counts()
    pie_descr = list(s.index)
    data      = [float(x) for x in list(s.values)]
    pie_descr = ["".join([str(int(x))," colums with ",y,
                 " (",str(np.round(x/np.sum(data)*100)),"%)"]) for x,y in zip(data, pie_descr)]

    # pie
    pie_size_scale =0.8
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5*pie_size_scale), radius=pie_size_scale,startangle=-45)
    
    # params for widgets
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    kw = dict(arrowprops=dict(arrowstyle="->"),
              bbox=bbox_props, zorder=0, va="center", fontsize=font_size)

    # add widgest to pie chart with pie descr
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))*pie_size_scale
        x = np.cos(np.deg2rad(ang))*pie_size_scale
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(pie_descr[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    ax.set_title(title)
    
    
    
    
    
    
    

    
# Function, ............................................................................
     
def table_with_stats_on_missing_non_missing_data_in_df(*, df, fig_size, fig_number):
    
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Plots image of a table with basic statitics on the amount 
                            of missing and non-missing data in large dataFrame
        
        Parameters/Input              
        _________________   _______________________________________________________________________________        
        * df.               DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * fig_size          tuple, (int, int)
        * fig_numer         fiugure number that will be added to that plot
        
        Returns               
        _________________   _______________________________________________________________________________
        
        * plt figure
    """    
        
    # data preparatio
    col_nr = int(df.shape[0])
    nr_of_columns_with_10_or_less_or_nan = (df.NaN_perc<=10).sum()
    nr_of_columns_with_11_to_25_perc_of_nan = df.loc[(df["NaN_perc"]>10) & (df["NaN_perc"]<=25)].shape[0]
    nr_of_columns_with_26_to_50_perc_of_nan = df.loc[(df["NaN_perc"]>25) & (df["NaN_perc"]<=50)].shape[0]
    nr_of_columns_with_51_to_75_perc_of_nan = df.loc[(df["NaN_perc"]>50) & (df["NaN_perc"]<=75)].shape[0]
    nr_of_columns_with_76_to_100_perc_of_nan = df.loc[(df["NaN_perc"]>75)].shape[0] 
    ###
    mean_nr_of_noNaN_per_row = str((df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0]).round(3))
    mean_perc_of_noNaN_per_row = str(((df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0])/col_nr*100).round(3))
    ###
    mean_nr_of_NaN_per_row = str((col_nr-df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0]).round(3))
    mean_perc_of_NaN_per_row = str(((col_nr-df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0])/col_nr*100).round(3))


    # collect all data for the table:
    dct_table_numbers={
    "Row number": str(int(df.nr_of_all_rows_in_original_df.mean().round(0))),
    "Column number": str(df.shape[0]),
    ".    " : str(""),  
    "non-missing data per Row": mean_nr_of_noNaN_per_row,
    "missing data per Row":  mean_nr_of_NaN_per_row,
    "....  " : str(""),        
    "Mean %/nr of non-missing data per column": str(int(df.nr_of_non_null_values.mean().round(0))),
    "Mean %/nr of missing data per column" : str(int(df.nr_of_all_rows_in_original_df.sum()-df.nr_of_non_null_values.sum())),
    "Mean %/nr of unique values per column" : str(int(df.nr_of_unique_values.mean().round(0))),
    "    " : str(""),       
    "- Columns with <=10% of NaN": nr_of_columns_with_10_or_less_or_nan,
    "- Columns with >10% NaN <= 25%":  nr_of_columns_with_11_to_25_perc_of_nan,
    "- Columns with >25% NaN <= 50%": nr_of_columns_with_26_to_50_perc_of_nan,    
    "- Columns with >50% NaN <= 75%": nr_of_columns_with_51_to_75_perc_of_nan,        
    "- Columns with >75% of NaN ": nr_of_columns_with_76_to_100_perc_of_nan
    }
    
    dct_table_perc={
    "Row number": "100%",
    "Column number": "100%",
    " " : str(""),
    "non-missing data per Row": "".join([mean_perc_of_noNaN_per_row,"%"]),
    "missing data per Row": "".join([mean_perc_of_NaN_per_row,"%"]),
    "....  " : str(""),    
    "Mean %/nr of non-missing data per column": "".join([str(((df.nr_of_non_null_values/df.nr_of_all_rows_in_original_df).mean()*100).round(1)),"%"]),
    "Totla %/nr of Missing data in DataFrame" : "".join([str((100-df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df.sum()*100).round(1)),"%"]),
    "Mean %/nr of of unique values per column" : "".join([str(((df.nr_of_unique_values/df.nr_of_all_rows_in_original_df).mean()*100).round(1)),"%"]),        
    "     " : str(""),
    "%/nr of Columns with <=10% of NaN": "".join([str(np.round((int(nr_of_columns_with_10_or_less_or_nan)/col_nr)*100,1)), "%"]),
    "%/nr of Columns with >10% NaN <= 25%": "".join([str(np.round((int(nr_of_columns_with_11_to_25_perc_of_nan)/col_nr)*100,1)), "%"]),
    "%/nr of Columns with >25% NaN <= 50%": "".join([str(np.round((int(nr_of_columns_with_26_to_50_perc_of_nan)/col_nr)*100,1)), "%"]),   
    "%/nr of Columns with >50% NaN <= 75%": "".join([str(np.round((int(nr_of_columns_with_51_to_75_perc_of_nan)/col_nr)*100,1)), "%"]),        
    "%/nr of Columns with >75% of NaN ": "".join([str(np.round((int(nr_of_columns_with_76_to_100_perc_of_nan)/col_nr)*100,1)), "%"]),
    } 
    
    # np.array with data so they can be displayed on plot
    arr = np.zeros([len(dct_table_numbers.keys()), 3], dtype="object"); arr
    arr[:,0] = list(dct_table_numbers.keys())
    arr[:,1] = list(dct_table_perc.values())
    arr[:,2] = list(dct_table_numbers.values())
    arr[(2,5,9),0] = [".", ".","."]
     
    # figure
    sns.set_context("notebook")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size, facecolor="white")
    fig.suptitle(f"Fig.{fig_number} Number or % of rows/cells with missing/non-missing data in entire DataFrame")
    
    # add table to ax
    table = plt.table(cellText=arr, colLabels=['Category',"%", "Number" ], loc='center', cellLoc='left', colColours=['lightgrey']*3)
    table.auto_set_column_width((-1, 0, 1, 2, 3))
    table.scale(1, 3)
    #table.auto_set_font_size(False) # numbers look better with autmatic fontsize setting and sns.set_style()
    #table.set_fontsize(10)
    
    # remove all plot spines, ticks and labels:
    ax.spines['top'].set_visible(False) # remove axis ...
    ax.spines['right'].set_visible(False) # ...
    ax.spines['left'].set_visible(False) #  ...
    ax.spines['bottom'].set_visible(False) #  ...
    ax.xaxis.set_ticks_position('none')# remove ticks ...
    ax.yaxis.set_ticks_position('none')# ...
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    
    # adjust
    fig.subplots_adjust(top=0.65)# to adjust for the title





    
    
    
    
    
# Function, ............................................................................
   
def barplot_with_data_completness_class_description_and_top_value_examples(*, data_examples, top_val_perc, df_filter, plot_title, 
                                                                           fig_size=(12,12), font_size=8, group_size=5):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Generate bar plot used to get fast information on data 
                            in different column in large df
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * data_examples     DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()
        
        .
        * df_filter         list, with True/False for each row in data_examples & top_val_perc 
                            if True, the row will be displayed on barplot
                            
        * fig_size          tuple, (row lenght, col lenght), in inches
        * font_size         int, size of all fonts used on the plot
        * group_size        int, how many rows will be diplayes as group on y axis on horizonal barplot.
                            groups are divided by space == to one bar.
                            
        Returns             
        _________________   _______________________________________________________________________________
        
        * plt.figure
        
        
    """
    
    
    #### names and group filtering

    # group names,
    group_names = list(data_examples.name[df_filter])

    # select data for plot,
    data_completness  = 100-np.array(data_examples.NaN_perc[df_filter]).flatten()
    tick_description  = data_examples.name[df_filter]
    top_values        = top_val_perc.values[df_filter, :]
    top_data_examples = data_examples.examples[df_filter]
    group_description = data_examples.summary[df_filter]

    # rescale top values,so they are part of non-missing data
    for i in range(top_values.shape[1]):
        v = top_values[:,i]
        top_values[:,i] = (v*data_completness)/100
    all_remaining_values = data_completness-top_values.sum(axis=1)

    # join the data in one array, I had some problems here, 
    data_for_plot = np.round(np.c_[(np.round(top_values,1), all_remaining_values)],1)

    
    
    #### order the bars,

    # find order of the bars, based on data completness,
    bar_order = pd.DataFrame(np.c_[(data_completness, np.arange(data_completness.size))]).sort_values(0, ascending=True)
    bar_order.reset_index(inplace=True, drop=True)
    bar_order = pd.concat([bar_order, pd.Series(list(bar_order.index))], axis=1)
    bar_order = bar_order.sort_values(1,ascending=True)
    bar_order = np.array(list(bar_order.index))

    # add spaces between everyx 5th bar, 
    add_spacers = True
    if add_spacers==True:
        # add spaces between everyx 5th bar, 
        space_between_groups = 1
        new_br = bar_order.copy().flatten()
        group_top  = []
        group_bottom = []

        for i, j in enumerate(sorted(list(bar_order))):

            if i==0: 
                add_to_list, counter = 0, 0
                group_bottom.append(j)

            if i>0 and counter<group_size: 
                counter +=1       

            if counter==group_size:
                group_bottom.append(j+add_to_list+1)
                counter=0
                add_to_list +=space_between_groups; 

            new_br[bar_order==j]=j+add_to_list

        group_top = [x+group_size-1 for x in group_bottom]    
        group_top[-1] = np.max(bar_order)+add_to_list
        bar_order = new_br.copy()

        
        
        
    #### barplot parameters; this was just to help me in long function !
    
    numeric_data_for_plot = data_for_plot # np array, 
    top_data_examples     = top_data_examples
    bar_position          = bar_order + 1
    group_description     = group_description
    bar_related_fontsize  = font_size
    

    
    #### bar_names, (ytick labels),
   
    if len(list(data_examples.dtype[df_filter].unique()))>1:
        df_bar_names = pd.DataFrame({"col_1":group_names, "col_2":list(data_examples.dtype[df_filter])}) 
        df_bar_names.col_2 = df_bar_names.col_2.str.pad(width=20, side="left", fillchar=".")
        bar_names = list(df_bar_names.col_1.str.cat([", "]*df_bar_names.shape[0]).str.cat(df_bar_names.col_2))
    else:
        bar_names             = group_names # list, old script, now chnaged as in below
        
        
        
    #### barplot,

    # imports
    from matplotlib import colors
    import matplotlib.patches as patches

    # helper,
    def stacked_barh_one_level(*, f_ax, bar_pos, top, bottom, colors, edgecolor, labels):
        f_ax.barh(bar_pos, top,left=bottom, color=colors, edgecolor=edgecolor, label=labels, linewidth=0.5, height=0.6)
        return f_ax

    # Set style and colors,
    plt.style.use("classic")
    bar_colors = plt.get_cmap("tab10")(np.linspace(0, 0.5, data_for_plot.shape[1])) # different nr of colors,
    edge_colors = bar_colors.copy()
    bar_colors[-1,0:3] = colors.to_rgb("lightgrey")
    edge_colors[-1,0:3] = colors.to_rgb("grey")

    # fig
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size, facecolor="white")
    fig.suptitle(plot_title)
    plot_top_value = np.max(bar_position)+15
    ax.set_ylim(0,plot_top_value)
    ax.set_xlim(0,300)

    # add top values as % od data completness,
    counter =[]
    add_top_values=True
    if add_top_values==True:
        counter = 0
        for i in list(range(data_for_plot.shape[1]))[::-1]:
            if counter == 0:
                bar_start = [0]*data_for_plot.shape[0]    
                bar_end   = data_for_plot[:,i]
            else:
                bar_start = bar_start+bar_end           
                bar_end   = data_for_plot[:,i] # bar end is hot tall is an individual bar, segment, not top point on a graph
            counter+=1

            # plot level on stacked plot
            ax = stacked_barh_one_level(
                 f_ax=ax, 
                 bar_pos=bar_position, top=bar_end, bottom=bar_start, 
                 colors=bar_colors[i], edgecolor=bar_colors[i], labels="test",
                 )

            
    #### ad legend on y axis        


    # Add ticks on y axis, and names for each bar,
    ax.set_yticks(bar_position)
    ax.set_yticklabels(bar_names, fontsize=bar_related_fontsize, color="black")
    ax.set_xticks([0, 25,50,75,100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=bar_related_fontsize, color="black")

    # Format ticks,
    ax.tick_params(axis='x', colors='black', direction='out', length=4, width=2) # tick only
    ax.tick_params(axis='y', colors='black', direction='out', length=4, width=2) # tick only    
    ax.yaxis.set_ticks_position('left')# shows only that
    ax.xaxis.set_ticks_position('bottom')# shows only that

    # Remove ticks, and axes that you dot'n want, format the other ones,
    ax.spines['top'].set_visible(False) # remove ...
    ax.spines['right'].set_visible(False) # remove ...  
    ax.spines['bottom'].set_linewidth(2) # x axis width
    ax.spines['bottom'].set_bounds(0,100) # Now the x axis do not go under the legend
    ax.spines['left'].set_linewidth(2) # y axis width 

    # Add vertical lines from grid,
    ax.xaxis.grid(color='grey', linestyle='--', linewidth=1) # horizontal lines

    # add patch on top to remove surplus gridlines
    x_left = -100
    rect_width = 500
    y_bottom = np.max(bar_position)+1.4 # add a bit to nut cut text opr boxes,
    rect_height = 500
    rect = patches.Rectangle(xy=(x_left,y_bottom),
                                 width=rect_width,height=rect_height,
                                 linewidth=0,edgecolor='white',facecolor='white',alpha=1, zorder=10)
    ax.add_patch(rect)
    plt.ylim(top=np.max(bar_position)+1.4)

    
    #### axes desciption    
    ax.set_xlabel("Percentage of non-missing data                                 ", ha="right") # I intentionally, added these spaces here!
    ax.set_ylabel("Column name and dtype name", ha="center")


    #### add, numbers and examples

    # add rectagles arrnoud examples
    for i, j in zip(group_bottom, group_top):
        x_left = 113
        rect_width = 186
        ###
        y_bottom = i+0.2
        rect_height = j-i+1.5
        ###
        rect = patches.Rectangle(xy=(x_left,y_bottom),
                                 width=rect_width,height=rect_height,
                                 linewidth=1,edgecolor="darkgreen",facecolor='yellow',alpha=0.3)
        ax.add_patch(rect)

    # add text with data completness above each bar,
    add_text_wiht_data_completness_above_each_bar=True
    if add_text_wiht_data_completness_above_each_bar==True:
        for i in range(numeric_data_for_plot.shape[0]):
            text_y_position = bar_position[i]-0.3
            text_x_position = numeric_data_for_plot.sum(axis=1).tolist()[i]+2

            # text,
            text_to_display = "".join([str(int(np.round(numeric_data_for_plot.sum(axis=1).tolist()[i],0))),"%"])
            t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=bar_related_fontsize, color="darkred")
            #t.set_bbox(dict(facecolor="white", alpha=0.3, edgecolor="white"))
    else: "do nothing"    

    # Add table, wiht data to plot,
    for i in range(numeric_data_for_plot.shape[0]):
        text_y_position = bar_position[i]-0.3
        text_x_position = 115

        # text,
        text_to_display = list(group_description)[i]
        t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=bar_related_fontsize, color="black")

    # add examples,   
    for i in range(numeric_data_for_plot.shape[0]):
        text_y_position = bar_position[i]-0.3
        text_x_position = 170

        # text,
        if re.search("all nonnull",str(list(group_description)[i])):
            text_to_display = "".join(["- - - > ",list(top_data_examples)[i]])
        else: text_to_display = list(top_data_examples)[i]
        t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=bar_related_fontsize+2, color="black")

    #### add plot legend  

    box_color       = "yellowgreen"
    box_edge_color  = "darkgreen"
    text_color      = "black" 
    text_size       = bar_related_fontsize

    text_x_position = 3
    text_y_position = np.max(bar_position)+2.5
    text_to_display = """Each Bar Shows Percentage of Non-missing Data In \n A Given Column, And The Three Most Frequent Values In \n That Column Shown As Different Colors On Top Of It"""
    t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=text_size, color=text_color, ha="left")
    t.set_bbox(dict(facecolor=box_color, alpha=1, edgecolor=box_edge_color))

    text_x_position = 115
    text_y_position = np.max(bar_position)+2.5
    text_to_display = """Data Description \n - numeric.: min; mean; max \n - string/time: nr of classes"""
    t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=text_size, color=text_color, ha="left")
    t.set_bbox(dict(facecolor=box_color, alpha=1, edgecolor=box_edge_color))

    text_x_position = 175
    text_y_position = np.max(bar_position)+2.5
    text_to_display = """The Three Most Frequent Non-Missing Values, Each in tuple, separated with ';', Each With:\n - a) percentage of rows, with non-missing data in that column with that particular value, \n - b) and, if it was a string, its first 15 characters, or the number """
    t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=text_size, color=text_color, ha="left")
    t.set_bbox(dict(facecolor=box_color, alpha=1, edgecolor=box_edge_color))
    
    fig.subplots_adjust(top=0.8)
    plt.show();
    



    


    

    
    
# Function, ............................................................................
       
def examine_df_visually(*, data_examples, top_values_perc, start_figure_numbers_at=1, groups_to_display="all", pieChart=True, showTable=True, barPlot=True):
 
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Plots Pie chart, table and barplot summarizing data in large dataFrame
        
        Parameters/Input              
        _________________   _______________________________________________________________________________  
        
        . Input .
        
        * data_examples     DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()
        * groups_to_display str, or list with strings, {"all", "text", "numeric", "datetime"}
                            "all", (default), or one of the dtypes, in data_examples.dtype, 
                            or list with different dtypes that will be ploted on rseraprate barplots
                            Columns only with missing data are not included in groups, these are plotted
                            only with "all" default option
                
        . Parameters . 
        
        * start_figure_numbers_at 
                            >=1, how nto start numeration of the figures with plots
        * pieChart          if True (default), display Pie chart with dtypes detected in data_examples
                            with number start_figure_numbers_at 
        * showTable         if True (default), display image of a summary table
                            with number start_figure_numbers_at + 1
        * barPlot.          if True (default), displays
                            with number start_figure_numbers_at  + 2, 3,4 and so on for each dtype
                            
                            
        Returns               
        _________________   _______________________________________________________________________________
        
        * Pie chart.        by :   pie_chart_with_perc_of_classes_df_column()
        * Tamble image.     by :   table_with_stats_on_missing_non_missing_data_in_df()
        * BarPlot           by :   barplot_with_data_completness_class_description_and_top_value_examples()
        
    """    

    figure_counter = 0
    
    # .. Pie plot to check nr of col with different dtypes in data df,
    if pieChart==True:
        sns.set_context("notebook")
        fig, axs = plt.subplots(nrows=1, ncols=1,
                                figsize=(4, 5), facecolor="white",
                               subplot_kw=dict(aspect="equal"))
        fig.suptitle(f"Fig.{figure_counter+start_figure_numbers_at} Data Types in different columns", fontsize=18)
        pie_chart_with_perc_of_classes_df_column(ax=axs, s=data_examples["dtype"], title="")
        plt.subplots_adjust(top=0.9)
        plt.show() 
        
        # add 1 to figure number
        figure_counter +=1
        
    if showTable==True:
        figNumber = figure_counter+start_figure_numbers_at
        table_with_stats_on_missing_non_missing_data_in_df(df=data_examples, fig_size=(5,5), fig_number=figNumber)
        
        # add 1 to figure number
        figure_counter +=1
        
        
    if barPlot==True:
        
        # .. barplot for each column with any non-missing data, eacg dtype is plotted separately,
        if groups_to_display=="all": groups_to_display=["all"]; add_all_groups=True
        else: add_all_groups=False
        
        for i, group_name in enumerate(groups_to_display): 
            
            # filter the data, and plot title, 
            if add_all_groups:
                df_filter         = pd.Series([True]*data_examples.shape[0])
                plot_title        = f"Fig.{start_figure_numbers_at+figure_counter} Summary for Columns with all data types, Caution: % values were rounded, and that may not sum up to 100" 
            
            else:
                df_filter         = data_examples['dtype']==group_name
                plot_title        = f"Fig.{start_figure_numbers_at+figure_counter} Summary for Columns with {group_name} data types, Caution: % values were rounded up" 

            # test, if the given group was present:  
            if df_filter.sum()==0:
                print("- - -THERE WERE NO COLUMNS WITH THAT DATA TYPE IN SEARCHED DataFrame - - -", end="\n\n")

            else:
                if df_filter.sum()>0 and df_filter.sum()<=10:
                    figSize = (12,5)
                    groupSize = df_filter.sum()

                if df_filter.sum()>10:
                    figSize = (12,12)
                    groupSize = 5  
                 
                if df_filter.sum()>50:
                    figSize = (12,22)
                    groupSize = 8  
                ##    
                barplot_with_data_completness_class_description_and_top_value_examples(    data_examples=data_examples, 
                                                                                           top_val_perc=top_values_perc, 
                                                                                           df_filter=df_filter, 
                                                                                           plot_title=plot_title,
                                                                                           fig_size=figSize,
                                                                                           font_size=8,
                                                                                           group_size=groupSize)
            # add 1 to figure number
            figure_counter +=1

            # example
            #groups_to_display = ['text', 'numeric', 'datetime']
            # plot_summary_pie_chart_and_bar_plots(data_examples=data_examples, top_val_perc=top_val_perc, start_figure_numbers_at=4, pieChart=True, showTable=True, groups_to_display = ['text', 'numeric', 'datetime'])







# Function, ...........................................................................................

def calculate_similarity_index(*, df, df_ex, select_in_df="all", df_row_perc=10, verbose=True, detailed_info_verbose=False):
    """ 

        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        Function            Perfomrs cross-validation of all or selected columns in df with any dtype 
                            (text, dattime and numeric), and returns similarity index for each comparison 
                            (0, 1]
        
                           
        Parameters/Input            
        _________________   _______________________________________________________________________________     

        . Input .
        * df                DataFrame, column names are stributes that will be cross-validate
        * df_ex             Summary DataFrame, for the above df, where each row, describes one column in the above df
        
        
        . Parameters .
        * select_in_df      {"all", "text", "datetime", "numeric"}, 
                            name of dtype in df, "all" by default
        * df_row_perc       "all", or int >0 & <100, percentage of randomly selected rows in df, used to reduce 
                            the size of compared 
        * display_messages  Bool, if True, progress messages on each comparison are diplayed

        Returns               
        _________________   _______________________________________________________________________________
        
        * DataFrame         with two statistics, col1_unique_el_nr/col1+col2_unique_el_nr and 
                            col2_unique_el_nr/col1+col2_unique_el_nr
                            names of compared columns in df, raw data, and number of unique records 
                            in each combination of columns

        Returned Values              
        _________________   _______________________________________________________________________________     

        * Similarity Index  if SI =1, unique items, suplicated in different columns are distributed in the same way
                                      ie. these columns may be a potential duplicates
                            if SI ~0, all groups of unique record in both columns are distributed randomly 
                            
        * NaN               NaN are removed in each combination of compared column, only if present in both columns

                            
    """
    

    # FIRST TEST:  check if df and df_ex contain the same data, 
    if df.shape[1]!=df_ex.shape[0]: 
        
        # .. YOU HAVE TO SPOT HERE AND PROVIDE df's from the same dataset, 
        print(f"\n\n - ERROR - please meke sure df and df_ex arer from the same dataset ! \n\n")
    
    else: 
        
        # .. YOU CONTINUE, and you may extract list with column names to cross-validate,  
        if select_in_df!="all": cols_to_crossvalidate = list(df_ex.name[(df_ex.summary!="all nonnull values are unique") & (df_ex.dtype==select_in_df)])
        else: cols_to_crossvalidate = list(df_ex.name[df_ex.summary!="all nonnull values are unique"])
            
            
        # .. SECOND TEST: check if you have at least two columns to compare,
        if len(cols_to_crossvalidate)<2:
            
            # .. YOU HAVE TO SPOT HERE because there is nothing to cross-validate ! 
            print(f"\n\n - ERROR - you have less then 2 columns to compare, maybe one of them had only unique information and was removed ! \n\n")
        
        # .. if yes continue and perform cross-validation, 
        else: 
            # .. YOU CONTINUE, and perfomr at least one comparison, each comparison is again evaluated, individually,
            
            
            # ....................................................................................................     
            # Extract sub-set of rows, if requested, this allow to speed up the process, 
                        
            # .... reduce df size, ie. row nr to speed up, with df_row_perc parameter, 
            if df_row_perc=="all": 
                "all rows will be used to calaulate SI"
            if df_row_perc==100: 
                "all rows will be used to calaulate SI"
            else:
                dfseq = list(range(df.shape[0]))
                row_list = random.sample(dfseq, k= int(df.shape[0]*df_row_perc/100))
                df = df.iloc[row_list,:] # done:


                
            # ....................................................................................................     
            # build object to store the results, and print info,
            
            # .... First, find all unieque combinations of indexes, of cols to crossvalidate, 
            n = len(cols_to_crossvalidate)
            a = list(range(n))
            b = list(range(n))
            combination_list = []
            for i, ai in enumerate(a):
                for j, bj in enumerate(b):
                    combination_list.append((ai, bj))
                    if j==len(b)-1: b.remove(ai) # to have only unique combinations            
            
            
            # .... create df to store results,
            cv_results_list = [0]*len(combination_list)
            cv_results_df = pd.DataFrame({"col1/col1col2":cv_results_list, "col2/col1col2":cv_results_list, "nr_of_compared_items":cv_results_list,
                                          "col1_name":cv_results_list, "col2_name":cv_results_list, "col1_grid_pos":cv_results_list, "col2_grid_pos":cv_results_list,
                                          "col1_class_nr":cv_results_list, "col2_class_nr":cv_results_list,"col1col2_class_nr":cv_results_list,
                                          "stat_mean":cv_results_list,"pcr5":cv_results_list})
            # .... and short info to display,
            if verbose==True:
                print(f"\n Performing Cross-validation of {cv_results_df.shape[0]} combinationas of columns in df ::::::::""")
            if verbose==False:
                print(f"calculating similarity index for {cv_results_df.shape[0]} unique combinations of columns in df . .", end=".")
        
        
            # ....................................................................................................     
            # Run cross-validation, for loop over it, 
                  
            # .... run for loop, ouver each comparison, 
            for i, cv_comb in enumerate(combination_list):    

                # ...... more detailed info to display,
                if detailed_info_verbose==True:
                    i_space = np.array([int(x) for x in list(np.ceil(np.linspace(0,len(combination_list), 10)).clip(0,len(combination_list)))])
                    i_space_names = pd.Series([ "".join([str(x),'%']) for x in list(range(0,101,10)) ])
                    print(f"{i}; ", end="")
                    if np.sum(i_space==i)>0: 
                        print(f"""{i_space_names.loc[list(i_space==i)].values[0]} eg: {cols_to_crossvalidate[cv_comb[0]]} vs {cols_to_crossvalidate[cv_comb[1]]} at {pd.to_datetime("now")}""",end='\n')

                   
                
                # ....................................................................................................     
                # Extract two columns and remove rows with nan in both of them,   
                    
                # ...... extract pair of compared columns, and put info into results df, 
                two_cols_df = df.loc[:,[cols_to_crossvalidate[cv_comb[0]], cols_to_crossvalidate[cv_comb[1]]]]
                two_cols_df.columns=["one", "two"]
                cv_results_df.iloc[i,[3,4]]=[cols_to_crossvalidate[cv_comb[0]], cols_to_crossvalidate[cv_comb[1]]]
                cv_results_df.iloc[i,[5,6]]=[cv_comb[0], cv_comb[1]]

                
                # ...... remove paris of NaN from compared columns, 
                two_cols_df = two_cols_df.dropna(how="all", axis=0)
                two_cols_df.reset_index(drop=True) 
                
                
                # ...... add number of compared items, 
                cv_results_df.iloc[i, 2] = two_cols_df.shape[0]

               
            
                # ....................................................................................................     
                # Test if you have anything left to wrk with,                
                
                # ...... continue if you have any data to compare,
                if two_cols_df.shape[0]<=1:
                    cv_results_df.iloc[i,[0,1]] = [np.nan, np.nan] # place nan if there is nothign to compare !
                    cv_results_df.iloc[i, 11] = np.nan
                    cv_results_df.iloc[i, 10] = np.nan
                    # done, no more work with that pair, 
                    
                else:
                    
                    # ....................................................................................................
                    # Find information require for calulating similarity index, 
                    
                    
                    # .......... start by replacing non-duplicated misssing data in each column, wiht some string, 
                    #            to not risk having zero groups!
                    two_cols_df.loc[two_cols_df.one.isna(), "one"]="NaGroup"
                    two_cols_df.loc[two_cols_df.two.isna(), "two"]="NaGroup"

                    # .......... count unique items in each column and in combined column, 
                    c1_unique_el    =  two_cols_df.groupby(["one"]).size().shape[0]
                    c2_unique_el    =  two_cols_df.groupby(["two"]).size().shape[0]
                    c1c2_unique_el  =  two_cols_df.groupby(["one", "two"]).size().shape[0]
                    cv_results_df.iloc[i,[7,8,9]]=[c1_unique_el, c2_unique_el, c1c2_unique_el]

                    
                    
                    # ....................................................................................................
                    # calculate similarity indexes, with different methods,
                               
                        
                    # +++++++++ similiraty index for each row in each combination, 
                    cv_results_df.iloc[i,[0,1]] = [c1_unique_el/c1c2_unique_el, c2_unique_el/c1c2_unique_el]
                    cv_results_df.iloc[i,[5,6]]=[cv_comb[0], cv_comb[1]]
                    
                    # +++++++++ mean,
                    cv_results_df.iloc[i,10] = (c1_unique_el/c1c2_unique_el + c2_unique_el/c1c2_unique_el)/2
                    
                    # ++++++++ Use proportional conflict redistribution rule no. 5 (PCR5) (Smarandache & Dezert, 2006) 
                    #          to combine two SI values, for each colummn combination, creates log distrib over two combined values,
                    A = float(c1_unique_el/c1c2_unique_el)
                    B = float(c2_unique_el/c1c2_unique_el)

                    if A+B==2: PM = 1 
                    if A>0 and B>0 and (A+B)<2: PM = (A*B) + ((A**2)*(1-B))/(A+(1-B)) + ((B**2)*(1-A))/(B+(1-A))
                    if A+B==0: PM = 0

                    cv_results_df.iloc[i,11] = PM

            # .... return,
            if verbose==False:
                print(f" DONE . . . ", end="\n") 
            return cv_results_df.copy()





        
        
        
        
        

# Function, ...........................................................................................

def plot_heatmap_with_cv_results(*,df, fig_size=(12,12), title_fontsize=20, axes_fontsize=8, method="pcr5"):
    
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Generate Traangle shaped Heatmap with similarity 
                            index calulated for dataframe columns
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        * df                DataFrame, returned by calculate_similarity_index()
        .
        * fig_size          tuple, (int, int)
        * title_fontsize    int, fontsize for plot titlke and axes labels
        * axes_fontsize     int, fontsize for tick labesl and labels of pixes on the plot
        * method            str, {"mean", "pcr5", "min", "max"}; what values to display on heatmap
                            . pcr5 - SI calulated with proportional conflict redistribution rule no. 5 (PCR5) (Smarandache & Dezert, 2006)¨
                            . mean - SI is the mean of two individual SI values found for each dataset.
                            . min  - lower individual SI
                            . max - upper individual SI
                            
        Returns             
        _________________   _______________________________________________________________________________
        
        * plt.figure
        * comments          no colorbar available in that version,
        
        
    """
 
    # ...............................................................................
    # format data,
    
    # .. select values to display
    if method=="pcr5": si_values = pd.Series(df.pcr5)
    if method=="mean": si_values = pd.Series(df.stat_mean)
    if method=="min":  si_values = pd.Series(df.iloc[:,[0,1]].min(axis=1))
    if method=="max":  si_values = pd.Series(df.iloc[:,[0,1]].max(axis=1))    
    
    # .. create array for the result, lower part will be empty because we did all comarisons just once, 
    arr_res = np.zeros((df.col1_grid_pos.max()+1, df.col2_grid_pos.max()+1))
    arr_res.fill(-1) # -1 so everything is now white on a heatpmap.

    # .. fill in arr_res with result, 
    for i in range(df.shape[0]):
        arr_res[int(df.col1_grid_pos[i]), int(df.col2_grid_pos[i])] = float(si_values[i])

    # .. reverse complement, to nicely position triangle heatmap, on upper left corner of the plot, 
    arr_res = arr_res[:, ::-1]    
  
    # .. ticklabels -  ie. find column names associasted with each index, 
    col_name_and_index_pos = df.groupby(['col1_name','col1_grid_pos']).size().reset_index().rename(columns={0:'count'}).iloc[:,[0,1]]
    col_name_and_index_pos = col_name_and_index_pos.sort_values(by="col1_grid_pos")

       
    # ...............................................................................
    # figure,
     
    # .. colors, 
    #.        on heatmap, all empty cells shoudl be white,
    #.        import matplotlib as mpl; my_cmap = mpl.colors.ListedColormap(['white', 'white', 'white', 'yellow', 'orange', "darkred"])
    #.        https://www.pluralsight.com/guides/customizing-colormaps
    bottom = plt.cm.get_cmap('YlOrRd', 128)
    top = plt.cm.get_cmap('binary', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    my_cmap = mpl.colors.ListedColormap(newcolors, name='binary_YlOrRd')

    # .. figure, 
    sns.set_context(None)
    fig, ax = plt.subplots(figsize=fig_size, facecolor="white")
    im = plt.imshow(arr_res, aspect = 1, origin="upper", interpolation='nearest', cmap=my_cmap)
    fig.suptitle("""Similarity Index - Modified Jaccard Similarity Index - (0,1]\nif SI==1, the classes in two compared columns are distributed in the same way\n if SI->0, the classes in two compared columns are ideally mixed with each other""", fontsize=title_fontsize)
 

    # ...............................................................................
    # ticks and other aestetics, 
    
    # .. build groups in tick positions to separate 
    ax.set_xticks(np.arange(col_name_and_index_pos.shape[0]))
    ax.set_yticks(np.arange(col_name_and_index_pos.shape[0]))

    # .. and label them with the respective list entries
    ax.set_xticklabels(col_name_and_index_pos.iloc[::-1,0], fontsize=axes_fontsize)
    ax.set_yticklabels(col_name_and_index_pos.iloc[:,0], fontsize=axes_fontsize)

    # .. select tick labels colots
    tick_pos = list(range(col_name_and_index_pos.shape[0]))
    tick_label_colors =[]
    step=5 # how many times each color will be repated
    for i in range(10000):
        tick_label_colors.extend(["black"]*step)
        tick_label_colors.extend(["red"]*step)
        #    tick_label_colors  = tick_label_colors[0:col_name_and_index_pos.shape[0]]
    tick_label_colors = tick_label_colors[0:len(tick_pos)] 
        #    I produces large number of color combinations, and now I am cutting it as it shoudl be,

    # .. modify ticklabels colors, 
    for xtick, ytick, xcolor, ycolor in zip(ax.get_xticklabels(), ax.get_yticklabels(), 
                                            tick_label_colors, tick_label_colors[::-1]):
        xtick.set_color(xcolor)
        ytick.set_color(ycolor)

    # .. Rotate the tick labels and set their alignment,
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", rotation_mode="anchor")

    # .. Remove ticks, and axes that you dot'n want, format the other ones,
    ax.spines['top'].set_visible(False) # remove ...
    ax.spines['right'].set_visible(False) # remove ...
    ax.yaxis.set_ticks_position('none')# shows only that
    ax.xaxis.set_ticks_position('none')# shows only that
     
        
    # ...............................................................................
    # add text annotation to selected cells, to not have problems with colorbar,  
    
    # .. Loop over data dimensions and create text annotations,
    for i in range(arr_res.shape[0]):
        for j in range(arr_res.shape[1]):
            if arr_res[i, j]<0.7: " do nothing"
            else:
                text = ax.text(j, i, np.round(arr_res[i, j],2), ha="center", va="center", color="white", fontsize=axes_fontsize, zorder=10)

    # .. spine description
    ax.set_xlabel("Column name", fontsize=title_fontsize )
    ax.set_ylabel("Column name", fontsize=title_fontsize )
                
                 
    # ...............................................................................
    # grid,  
    
    # .. add, pseudogrid from lines, 
    for xy in range(arr_res.shape[0]+1):
        plt.axvline(x=xy+0.5, color="lightgrey", linestyle="-" )
        plt.axhline(y=xy+0.5, color="lightgrey", linestyle="-" )      

    # .. finally add, THICKED pseudogrid from lines and DOTS, to separate differently colored ticklabels,  
    for xy in range(-1,arr_res.shape[0]+1,5):
        lcolor = "black"
        plt.axvline(x=xy+0.5, color="orange", linestyle="-", linewidth=1)
        plt.axvline(x=xy+0.5, color=lcolor, linestyle=":", linewidth=1)
        #
        plt.axhline(y=xy+0.5, color="orange", linestyle="-", linewidth=1) 
        plt.axhline(y=xy+0.5, color=lcolor, linestyle=":", linewidth=1)  
  
    
    # ...............................................................................
    # show,  
    
    # show the figure
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()





# Function, ...........................................................................................
 
def show_similar_columns(*, df_summary, df_cv_results, SI_threshold=0.9):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          function that uses Cross-Valiadation results and data_examples
                            to display columns/attribtes of Dataframe that are potentially duplicated
        
        Parameters/Input              
        _________________   _______________________________________________________________________________        
        * df_summary        DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * df_cv_results     DataFrame with Similarity Index returned by 
                            calculate_similarity_index()
        
        * SI_threshold      SI cutoff, it works on mean SI calulated with both compared columns in each pair
                            
        
        Returns               
        _________________   _______________________________________________________________________________
        
        * DataFrame         non-identical with column pairs, with SI>=SI_threshold
        
        
    """    
      
    # extract relevant info
    similar_cols_df = df_cv_results.loc[df_cv_results.stat_mean >= SI_threshold, ["col1_name", "col2_name", "pcr5"]]
    similar_cols_df = similar_cols_df.loc[(similar_cols_df.col1_name==similar_cols_df.col2_name)==False,].sort_values("pcr5", ascending=False)
    similar_cols_df.reset_index(drop=True, inplace=True)

    # add examples
    for i in range(len(similar_cols_df)):
        ex1 = df_summary.loc[(df_summary.name==similar_cols_df.col1_name.iloc[i]),"examples"]
        ex2 = df_summary.loc[(df_summary.name==similar_cols_df.col2_name.iloc[i]),"examples"]
        if i==0:
            ex1list = [ex1] 
            ex2list = [ex2]
        else:
            ex1list.append(ex1)
            ex2list.append(ex2)

    # add exampes to df,
    similar_cols_df = pd.concat([similar_cols_df , pd.Series(ex1list), pd.Series(ex2list)], axis=1, sort=False) 
    similar_cols_df.columns=["attribute 1", "attribute 2", "similarity index", "examples attribute 1", "examples attribute 2"]

    # return
    return similar_cols_df