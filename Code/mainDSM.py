#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import csv
import time
import unicodedata
import xlsxwriter
import numpy as np
import pandas as pd
import tensorflow as tf
from modeldnnDSM import *

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Reshape, Flatten, Masking, Dense, Input, Dropout, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Layer, Masking, Lambda, Permute, TimeDistributed
from tensorflow.keras.layers import concatenate, multiply, subtract, Attention 
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

from tensorflow.python.keras.backend import get_session


import tempfile
import shutil
from tempfile import mkdtemp
import os.path as path
from sklearn import metrics

print("Tensorflow {} loaded".format(tf.__version__))
print("Numpy {} loaded".format(np.__version__))

    
def display_metrics(aux1, aux2, Y1, Y2, learn_Y1, timer, method, bidirectional, 
                    shortcut, maxpooling, selfattention, alignment, onlyconcat, training_instances):
    
    num_true_predicted_true = 0.0
    num_true_predicted_false = 0.0
    num_false_predicted_true = 0.0
    num_false_predicted_false = 0.0
    
    print( "Matching records...")
    if learn_Y1:
        real = list(Y1) + list(Y2) #true values
        predicted = list(aux2) + list(aux1) #aux2 learns Y1 and aux1 learns Y2
    else:
        real = list(Y2) #true values
        predicted = list(aux1)
    #print(real)
    #print(len(real))
    #print(predicted)
    #print(len(predicted))
    size_real = len(real)
    size_predicted = len(predicted)
    
    #Metrics
    print( "Metrics...")
    conf_matrix = 0; auc = 0 
    #C(i,j) is equal to the number of observations known to be in group i and predicted to be in group j.
    # 0 is negative and 1 is positive
    #The count of true negatives is C(0,0), false negatives is C(1,0), true positives is C(1,1)  and false positives 
    #is C(0,1).
    conf_matrix = metrics.confusion_matrix(real, predicted)
    auc = metrics.roc_auc_score(real, predicted) 
    accuracy_score = metrics.accuracy_score(real, predicted)
    precision_score = metrics.precision_score(real, predicted)
    recall_score = metrics.recall_score(real, predicted)
    f1_score = metrics.f1_score(real, predicted)
    #metrics.plot_confusion_matrix(predicted, real)
    #metrics.plot_roc_curve(predicted, real)
    
    
    #create workbook and worksheet
    outWorkbook = xlsxwriter.Workbook("dataset-dnn-accuracy.xlsx")
    outSheet = outWorkbook.add_worksheet("Results")
    outSheet.write("A1", "Index") #A1 is (0,0) (x,y)
    outSheet.write("B1", "Real") # is (0,1) (x,y)
    outSheet.write("C1", "Predicted") # is (0,2) (x,y)
    
    #Create file
    #file = open("dataset-dnn-accuracy","w+")
    
    for pos in range(size_real):
        outSheet.write(pos+1,0,pos)
        if float(real[pos]) == 1.0:
            outSheet.write(pos+1,1,"True")
            if float(predicted[pos]) == 1.0:
                num_true_predicted_true += 1.0
                #file.write(str(pos) + "\tTRUE\tTRUE\n")
                outSheet.write(pos+1,2,"True")
            else:
                num_true_predicted_false += 1.0
                #file.write(str(pos) + "\tTRUE\tFALSE\n")
                outSheet.write(pos+1,2,"False")
        else:
            outSheet.write(pos+1,1,"False")
            if float(predicted[pos]) == 1.0:
                num_false_predicted_true += 1.0
                #file.write(str(pos) + "\tFALSE\tTRUE\n")
                outSheet.write(pos+1,2,"True")
            else:
                num_false_predicted_false += 1.0
                #file.write(str(pos) + "\tFALSE\tFALSE\n")
                outSheet.write(pos+1,2,"False")

    #timer = (timer / float(int(size_real))) * 50000.0
    timer = (timer / size_real) * 50000.0
    print ("True-True =", num_true_predicted_true)
    print ("True-False =", num_true_predicted_false)
    print ("False-True =", num_false_predicted_true)
    print ("False-False =", num_false_predicted_false)
    acc_num = num_true_predicted_true + num_false_predicted_false
    pre_num = num_true_predicted_true
    pre_denom = num_true_predicted_true + num_false_predicted_true
    rec_num = num_true_predicted_true
    rec_denom = num_true_predicted_true + num_true_predicted_false
    acc = func_divide_by_zero(acc_num, size_real)
    pre = func_divide_by_zero(pre_num, pre_denom)
    rec = func_divide_by_zero(rec_num, rec_denom)
    f1_num = pre * rec
    f1_denom = pre + rec
    f1 = 2.0 * func_divide_by_zero(f1_num, f1_denom)
    
    #file.close()
    outWorkbook.close()
    print ("Accuracy =", acc)
    print ("Precision =", pre)
    print ("Recall =", rec)
    print ("F1 =", f1)
     
    
    print ("Metric = Deep Neural Net Classifier :", method.upper())
    print ("Bidirectional :", bidirectional)
    print ("Shortcut Connections:", shortcut)
    print ("Maxpolling :", maxpooling)
    print ("Inner Attention :", selfattention)
    print ("Hard Allignment Attention :", alignment)
    print ("Only Concatenation :", onlyconcat)
    print("Confusion Matrix =", conf_matrix)
    print ("AUC =", auc)
    print ("Accuracy-score =", accuracy_score)
    print ("Precision-score =", precision_score)
    print ("Recall-score =", recall_score)
    print ("F1-score =", f1_score)
    print ("Processing time per 50K records =", timer)
    print ("Number of training instances =", training_instances)
    print ("")
    

def fill_table_input(variable, chars, char_labels, max_seq_len, method, file_name):
    XA1 = variable
    #print(len(chars))
    shape=(len(variable), max_seq_len, len(chars))
    #print(shape)
    
    #vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/temporary-file-dnn-1-"
    #aux1 = np.memmap("temporary-file-dnn-1-" + method, mode="w+", shape=(len(XA1), max_seq_len, len(chars)), dtype=np.bool)
    dirpath = mkdtemp()
    vp = path.join(dirpath, file_name)
    #with tempfile.TemporaryDirectory() as tmp_dir:
    #tmp_dir = tempfile.TemporaryDirectory()
    #vp = path.join(tmp_dir, file_name)
    #vp = path.join(tmp_dir.name, file_name)
    vpm = vp + method
    aux1 = np.memmap(vpm, mode="w+", shape=(len(variable), max_seq_len, len(chars)), dtype=np.bool)
    #print(type(aux1))
    #print(aux1)
    
    enum_variable = enumerate(variable)
    print(enum_variable)
    print(next(enum_variable))
    print(char_labels)
    
    #i = len(variable)
    #t = max_seq_len
    #char_labels[char] = len(chars)
    for i, example in enum_variable:
        print("example =", example)
        for t, char in enumerate(example):
            print("t =", t)
            print("char =", char) # key
            print(char_labels[char]) #value
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    #files_in_dir = os.listdir(tmp_dir.name)
    temp = 3
    #files_in_dir = os.listdir(vp + method) #need the right path
    #shutil.rmtree(dirpath) #file is still being used
    #shutil.rmtree(vpm)
      
    return aux1, dirpath


def string_embedding(dataset):
    num_true = 0.0
    num_false = 0.0
        
    with open(dataset, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res"], delimiter='|')                
        row_count = 0
        for row in reader:
            row_count +=1
        print(row_count)     
                
    print(num_false);
    XA1 = []
    XB1 = []
    XC1 = []
    Y1 = []
    XA2 = []
    XB2 = []
    XC2 = []
    Y2 = []
    print(type(Y1))
    #row_count = num_true + num_false
    #mid = (num_true + num_false) / 2.0
    training_instances = int(training_percent*row_count)
    mid = training_instances
    print("mid = " + str(mid))
    
    start_time = time.time()
    print( "Reading dataset... " + str(start_time - start_time))
    with open(dataset, encoding='utf-8') as csvfile:
        #reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "id1", "id2", "lat1", "lon1", "lat2", "lon2", "s3"], delimiter='|')
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res"], delimiter='|') 
        match_train_row_count = 0; match_test_row_count = 0; unmatch_train_row_count = 0; unmatch_test_row_count = 0
        
        print(reader.fieldnames)
        print(reader)
        
        start_time = time.time()
        for row in reader:
            print (row)
            print(row['res'])
           #print("res = " + row['res'])
            if row['res'] == "1": # match
                if len(Y1) < (mid):
                    Y1.append(1) # train
                    match_train_row_count +=1
                else:
                    Y2.append(1) # test
                    match_test_row_count +=1
            else:
                if len(Y1) < (mid):
                    Y1.append(0)
                    unmatch_train_row_count +=1
                else:
                    Y2.append(0)
                    unmatch_test_row_count +=1
                    
            print("s1 = " + row['s1'])
            #print("s2 = " + row['s2'])
            #row['s1'] = row['s1'] #.decode('utf-8')
            #row['s2'] = row['s2'] #.decode('utf-8')
            row['s1'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s1'] + u'|')), encoding='utf-8')
            row['s2'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s2'] + u'|')), encoding='utf-8')
            print("s1 = " + str(row['s1']))
            print("s2 = " + str(row['s2']))
            print("XA1 = " + str(XA1))
            if len(XA1) < (mid): # training
                XA1.append(row['s1'])
                XB1.append(row['s2'])
            else:
                XA2.append(row['s1'])
                XB2.append(row['s2'])
        print( "Dataset read... " + str(time.time() - start_time))
        
        print("match train row count = " + str(match_train_row_count))
        print("match test row count = " + str(match_test_row_count))
        print("unmatch train row count = " + str(unmatch_train_row_count))
        print("unmatch test row count = " + str(unmatch_test_row_count))
        
    print(Y1)
    print(Y1.count(1))
    print(Y2.count(1))
    
    Y1 = np.array(Y1, dtype=np.bool)
    Y2 = np.array(Y2, dtype=np.bool)
    print("XA1 = " + str(XA1))
    print("XA2 = " + str(XA2))
    print("XC1 = " + str(XC1))
    print("XC2 = " + str(XC2))
    list_char = XA1 + XB1 + XC1 + XA2 + XB2 + XC2
    print(type(list_char))
    print(list_char)
    
    '''
    temp_val = []
    for sublist in list_char:
        print(sublist)
        for val in sublist:
            temp_val.append(val)
    '''
    
    numerasised_list = [val for sublist in list_char for val in sublist]
    #numerasised_list = list([val for sublist in list_char for val in sublist]) 
    print(type(numerasised_list))
    print(numerasised_list)
    create_set = set(numerasised_list)
    print(create_set)
    chars = list(create_set)
    #chars = list(set(list([val for sublist in XA1 + XB1 + XC1 + XA2 + XB2 + XC2 for val in sublist])))
    print(chars)
    enum_chars = enumerate(chars)
    print(type(enum_chars))
    print(next(enum_chars))
    #for i,j in enum_chars:
    #    print(i,j)
    char_len = len(chars)    
    
    # Create a dictionary with order (item,index)
    char_labels = {ch: i for i, ch in enumerate(chars)}
    #print(char_labels)
    #print(len(XA1))
    #print(XA1)
   
    # Create Fixed Tables
    
    file_name = 'newfile1.dat'
    aux1 = fill_table_input(XA1, chars, char_labels, max_seq_len, rnn_method, file_name)
    print(XA1)
    #print(aux1)
    files_in_dir = os.listdir(aux1[1])
    XA1 = aux1
    del aux1
    print(XA1)
    files_in_dir = os.listdir(XA1[1])
   
    file_name = 'newfile2.dat'
    aux1 = fill_table_input(XB1, chars, char_labels, max_seq_len, rnn_method, file_name)
    XB1 = aux1
    del aux1
    
    file_name = 'newfile3.dat'
    aux1 = fill_table_input(XA2, chars, char_labels, max_seq_len, rnn_method, file_name)
    XA2 = aux1
    del aux1
    
    file_name = 'newfile4.dat'
    aux1 = fill_table_input(XB2, chars, char_labels, max_seq_len, rnn_method, file_name)
    XB2 = aux1
    del aux1
    print ("Temporary files created... " + str(time.time() - start_time))
    print ("Training classifiers...")

    if training_instances <= 0: training_instances = min(len(Y1), len(Y2))
    print(training_instances)
    print(type(XA1))
    print(XA1)
    
    return XA1, XB1, Y1, XA2, XB2, Y2, max_seq_len, char_len, training_instances 


def evaluate_string_dnn(dataset='dataset-string-similarity.txt', max_seq_len = 40, training_percent=0.7, num_epochs = 2, 
                        rnn_method='gru', bidir_num_layers=2, bidirectional=True, bidir_hiddenunits=60, 
                        hnn_maxpooling=False, hnn_num_layers=3, hnn_method = 'maxpooling1D',
                        selfattention=False , attention_method = 'softmax', maxpooling=False , 
                        alignment = False, shortcut=True, dnn_num_layers=2, dnn_hiddenunits=30, onlyconcat=False):
    
    
    XA1, XB1, Y1, XA2, XB2, Y2, max_seq_len, char_len, training_instances = string_embedding(dataset)
   
    learn_Y1 = False
    aux1, time1 = 0.0, 0.0
    aux2, time2 = 0.0, 0.0

    #Learn Y2
    aux1, time1 = deep_neural_net_rnn(train_data_1=XA1[0],
                                      train_data_2=XB1[0],
                                      train_labels=Y1, 
                                      test_data_1=XA2[0], test_data_2=XB2[0], test_labels=Y2, 
                                      max_len=max_seq_len, len_chars=char_len,
                                      bidir_num_layers = bidir_num_layers, bidirectional=bidirectional, rnn_method = rnn_method, bidir_hidden_units=bidir_hiddenunits, 
                                      hnn_maxpooling=hnn_maxpooling, hnn_num_layers=hnn_num_layers, hnn_method=hnn_method,
                                      selfattention=selfattention, attention_method = attention_method, maxpooling=maxpooling, 
                                      alignment = alignment , shortcut=shortcut, 
                                      dnn_num_layers = dnn_num_layers, dnn_hidden_units = dnn_hiddenunits, 
                                      onlyconcat=onlyconcat, num_epochs = num_epochs, n=1)
    
    if learn_Y1:
        #Learn Y1
        aux2, time2 = deep_neural_net_rnn(train_data_1=XA2[0],
                                          train_data_2=XB2[0],
                                          train_labels=Y2, 
                                          test_data_1=XA1[0], test_data_2=XB1[0], test_labels=Y1, 
                                          max_len=max_seq_len, len_chars=char_len,
                                          bidir_num_layers = bidir_num_layers, bidirectional=bidirectional, rnn_method = rnn_method, bidir_hidden_units=bidir_hiddenunits, 
                                          hnn_maxpooling=hnn_maxpooling, hnn_num_layers=hnn_num_layers, hnn_method=hnn_method,
                                          selfattention=selfattention, attention_method = attention_method, maxpooling=maxpooling, 
                                          alignment = alignment , shortcut=shortcut, 
                                          dnn_num_layers = dnn_num_layers, dnn_hidden_units = dnn_hiddenunits, 
                                          onlyconcat=onlyconcat, num_epochs = num_epochs, n=2)
    
    files_in_dir = os.listdir(XA1[1])
    files_in_dir = os.listdir(XB1[1])
    del XA1
    del XB1
    del XA2
    del XB2
   
    timer = time1 + time2
    print( "Total Time :", timer)
    
    display_metrics(aux1, aux2, Y1, Y2, learn_Y1, timer, rnn_method, bidirectional, 
                    shortcut, maxpooling, selfattention, alignment, onlyconcat, training_instances)
    sys.stdout.flush()
    
    

###################################################################
# MAIN
###################################################################    

#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/dataset_final_historical_places_distance.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/dataset_person_attributes1.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/Entity_matching_pipe2.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/Entity_matching_impure.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/quora_questions.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/restaurant_final.csv"
vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/amazon_walmart.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/scholar.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/amazon_google.csv"
#vp = "C:/Daniel/Code/String Matching/DeepLearning-StringMatching-master/datasets/buy.csv"

#Run model
max_seq_len = 300 # max length of sequence, space matter
training_percent = 0.7
num_epochs = 10

# Encoder
rnn_method = 'lstm' # 'lstm' #'gru' 'SimpleRNN'
bidir_num_layers = 5
bidirectional = True 
bidir_hiddenunits = 60 #60 
onlyconcat = False #exclude multiplication and substraction 
shortcut = False  

#Hierachical network
hnn_num_layers = 3 
hnn_maxpooling = False
hnn_method = 'globalmaxpooling1D' #'maxpooling1D' 'globalmaxpooling1D'

#Can only have alignment or selfattention or maxpooling 
alignment = False #False   
maxpooling = True #False
selfattention = False
attention_method = 'identity' #'softmax' 'proba' 'uniform' 'identity'

# Decoder
dnn_num_layers = 3
dnn_hiddenunits = 60 #60

  
evaluate_string_dnn(dataset=vp, max_seq_len = max_seq_len, training_percent=training_percent, num_epochs = num_epochs,
                         rnn_method = rnn_method, bidir_num_layers = bidir_num_layers, bidirectional = bidirectional, bidir_hiddenunits = bidir_hiddenunits, 
                         hnn_maxpooling = hnn_maxpooling, hnn_num_layers = hnn_num_layers, hnn_method = hnn_method,
                         selfattention = selfattention, attention_method = attention_method,
                         maxpooling = maxpooling, alignment = alignment, shortcut = shortcut, dnn_num_layers = dnn_num_layers, dnn_hiddenunits = dnn_hiddenunits, onlyconcat = onlyconcat)
