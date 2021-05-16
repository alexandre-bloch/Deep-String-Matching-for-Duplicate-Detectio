import pandas as pd
import numpy as np


""" 
For the Amazon-Google, Amazon-Walmart and Citation datasets
"""
################## Import Tables ########################################
tableA = pd.read_csv('C:/Users/alexa/Desktop/Dad/ZZZZ/NewData/Scholar/tableA.csv')
tableB = pd.read_csv('C:/Users/alexa/Desktop/Dad/ZZZZ/NewData/Scholar/tableB.csv')
match_train = pd.read_csv('C:/Users/alexa/Desktop/Dad/ZZZZ/NewData/Scholar/test.csv')
match_test = pd.read_csv('C:/Users/alexa/Desktop/Dad/ZZZZ/NewData/Scholar/train.csv')
match_valid = pd.read_csv('C:/Users/alexa/Desktop/Dad/ZZZZ/NewData/Scholar/valid.csv')
###########################################################################

################## Get a table of matche/unmatch ########################
match = pd.concat([match_train,match_test,match_valid],ignore_index=True)
match = match.drop_duplicates()

match.sort_values(by=['label'], ascending = False, ignore_index=True, inplace = True)
match_count = match['label'].value_counts()[1]
#########################################################################

################### Empty DF #############################################
index = []
for i in range(len(match)):
    index.append(i)
    
df = pd.DataFrame(index =index , columns=['col1'])
##########################################################################

################### Columns #############################################
a_cols = tableA.columns
b_cols = tableB.columns
match_cols = match.columns


for i in match.index:
    
    left = match['ltable_id'][i]
    right = match['rtable_id'][i]
    
    row_A = tableA.iloc[left]
    row_B = tableB.iloc[right]
    
    list_row_A = row_A.tolist()
    list_row_B = row_B.tolist()
    
    string_row_A = ' '.join(str(e) for e in list_row_A[1:])
    string_row_B = ' '.join(str(e) for e in list_row_B[1:])
    
    row = string_row_A + "|" + string_row_B
    
    if match['label'][i] == 1:
        row = row + "|" + "1"
    else:
        row = row + "|" + "0"    
    
    df['col1'][i] = row
###################################################################################
    
###################### Drop Weird Stuff ###########################################
    
df.replace(',','', regex=True, inplace=True)
df.replace('"','', regex=True, inplace=True)

df.col1.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
####################################################################################

result = df[0:round(3*match_count,-3)]

result = result.sample(frac=1)

result.to_csv('scholar.csv', index = False, header = False)


