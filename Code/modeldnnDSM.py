#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import csv
import time
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
from layersDSM import *


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Reshape, Flatten, Masking, Dense, Input, Dropout, SimpleRNN, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Layer, Masking, Lambda, Permute, TimeDistributed
from tensorflow.keras.layers import concatenate, multiply, subtract, Attention 
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

from tensorflow.python.keras.backend import get_session


import tempfile
import shutil
from tempfile import mkdtemp
import os.path as path
from sklearn import metrics


def set_gru_lstm_maxpooling_encoder(num_layers, gru, mask): 
    
    g1 = []
    g1_temp = gru[0](mask)
    print(g1_temp)
    g1_temp = Dropout(0.01)(g1_temp)
    g1_temp = MaxPooling1DMasked(pool_size=1, name = 'maxpooling1')(g1_temp)
    print(g1_temp)
    g1.append(g1_temp)

    num_size = num_layers
    for i in range(1,num_size):
        print(i)
        vp = 'maxpooling' + str(i+1)
        input_concat = concatenate([g1[i-1], mask])
        print(input_concat)
        g1_temp = gru[i](input_concat)
        print(g1_temp)
        g1_temp = Dropout(0.01)(g1_temp) #10% drop  
        g1_temp = MaxPooling1DMasked(pool_size=1, name = vp)(g1_temp)
        g1.append(g1_temp)
    
    final_input = concatenate(g1, axis=1)
    print(final_input)
    return final_input

#Can only have alignment or selfattention or maxpooling 
def define_gru_lstm_layers(num_layers, method, hidden_units, bidirectional, hnn_maxpooling=False, 
                           selfattention=False, maxpooling=False, alignment=False): 
    gru = []
    #gru = np.array([])
    
    if hnn_maxpooling:
        num_size = num_layers
    else:
        num_size = num_layers-1 # exclude last RNN
    print(num_size)
    for i in range(num_size):
        vp = method + str(i+1)
        if method =='gru':
            gru.append(GRU(hidden_units, implementation=2, return_sequences=True, name=vp))
        elif method =='lstm':
            gru.append(LSTM(hidden_units, implementation=2, return_sequences=True, name=vp))
            #np.append(gru, LSTM(hidden_units, implementation=2, return_sequences=True, name=vp))
            #tf.concat([gru, LSTM(hidden_units, implementation=2, return_sequences=True, name=vp)], 0)
        elif method =='simpleRNN':
            gru.append(SimpleRNN(hidden_units, return_sequences=True, name=vp))
        print(gru[i])
        
            
    if not hnn_maxpooling:    
        vp = method + str(num_layers)
        #temp_bool = (selfattention or maxpooling)
        temp_bool = (alignment or selfattention or maxpooling)
        if method =='gru':
            gru.append(GRU(hidden_units, implementation=2, return_sequences=temp_bool, name=vp))
        elif method =='lstm':
            gru.append(LSTM(hidden_units, implementation=2, return_sequences=temp_bool, name=vp))
            #np.append(gru, LSTM(hidden_units, implementation=2, return_sequences=temp_bool, name=vp))
        elif method =='simpleRNN':
            gru.append(SimpleRNN(hidden_units, return_sequences=temp_bool, name=vp))
        print(gru[num_size])
        
    
    #get_tensor_shape(gru[0])
        
    if bidirectional:
        bigru = []
        #bigru = np.array([])
        for j in range(num_layers):
            vp = "bi" + method + str(j+1)
            bigru.append(Bidirectional(gru[j], name=vp) )
            #np.append(bigru, Bidirectional(gru[j], name=vp) )
        #bigru = tf.convert_to_tensor(bigru)
        #bigru = tf.convert_to_tensor(bigru) 
        #return K.stack(bigru)
        return bigru
    else:
        return gru
    
       
def set_gru_lstm_encoder(num_layers, gru, mask, selfattention=False, 
                         attention_method = 'softmax', maxpooling=False, shortcut=False):       
    g1 = []
    num_size = num_layers-1 # exclude the last RNN
    print(num_size)
    
    #gru(mask)
    #test3 = array(gru)
    #test4 = test3[0](mask)
    
    for i in range(num_size):
        print(i)
        g1_temp = gru[i](mask)
        print(g1_temp)
        g1_temp = Dropout(0.01)(g1_temp) #10% drop   
        print(g1_temp)
        g1.append(g1_temp)
        
    # can have shortcut connections with selfattention or maxpooling
    last_gru = g1[num_size-1]
    print(last_gru)
    time_gru, unit_gru = get_tensor_shape(last_gru)
    if shortcut: # shortcut connections
        shortcut_con = concatenate([last_gru, mask])
        g2 = gru[num_size](shortcut_con) 
    else:
        g2 = gru[num_size](last_gru)
    g2 = Dropout(0.01)(g2)
    print(type(g2))
    time_gru, unit_gru = get_tensor_shape(g2)
    if selfattention: # selfattention
        #g2 = Attention()(g2) #need to be defined [query, value]
        g2 = SelfAttLayer()(g2)
        #g2 = Attention0()(g2)
        #g2 = AttentionLayer()(g2)
    elif maxpooling: # maxpooling
        g2 = GlobalMaxPooling1DMasked(name = 'maxpooling')(g2) #reshape 3D array into 2D array
        #g2 = GlobalMaxPooling1D(name = 'maxpooling')(g2)
        #print(g2.__len__)
       
    print(g2)
    time_gru, unit_gru = get_tensor_shape(g2)
            
    return g2

def set_deep_network_maxpooling_layers(dnn_num_layers, dnn_hidden_units, input_val):
    dense = []
    dropout = input_val
    num_size = dnn_num_layers-1 # exclude the last layer
    
    dense1 = Dense(dnn_hidden_units, activation='relu', name='dense1')(input_val)
    print(dense1)
    flatten = Flatten()(dense1)
    dropout = Dropout(0.01)(flatten)
    print(dropout)
    dense.append(dropout)
    
    if dnn_num_layers > 2:
        for i in range(1,num_size):
            vp = "dense" + str(i+1)
            dense_temp = Dense(dnn_hidden_units, activation='relu', name=vp)(dropout)
            dropout = Dropout(0.01)(dense_temp)
            print(dropout)
            dense.append(dropout)
            
        last_dense = dense[num_size-1]
        print(last_dense)
        time_gru, unit_gru = get_tensor_shape(last_dense)
    else:
        last_dense = dense[0]
        
    dense_output = Dense(1, activation='sigmoid', name='denseOutput')(last_dense) #1 output
    time_gru, unit_gru = get_tensor_shape(dense_output)
    
    return dense_output


def set_deep_network(dnn_num_layers, dnn_hidden_units, input_val):
    dense = []
    dropout = input_val
    num_size = dnn_num_layers-1
    for i in range(num_size):
        vp = "dense" + str(i+1)
        dense_temp = Dense(dnn_hidden_units, activation='relu', name=vp)(dropout)
        dropout = Dropout(0.01)(dense_temp)
        dense.append(dropout)
        
    last_dense = dense[num_size-1]
    print(last_dense)
    time_gru, unit_gru = get_tensor_shape(last_dense)
    
    dense_output = Dense(1, activation='sigmoid', name='denseOutput')(last_dense) #1 output
    time_gru, unit_gru = get_tensor_shape(dense_output)
    
    return dense_output
    

def deep_neural_net_rnn(train_data_1, train_data_2, train_labels, test_data_1, test_data_2, test_labels, max_len,
                        len_chars,  bidir_num_layers, bidirectional, method, bidir_hidden_units, 
                        hnn_maxpooling, hnn_num_layers, 
                        selfattention , attention_method, maxpooling, alignment, shortcut, 
                        dnn_num_layers, dnn_hidden_units, onlyconcat, num_epochs, n):
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    checkpoint_filepath = "checkpoint" + str(n) +".hdf5" #n is used
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, verbose=1, save_best_only=True) 
    
    #Define the GRU/LSTM layers
    if hnn_maxpooling:
        num_layers = hnn_num_layers
    else:
        num_layers = bidir_num_layers
    gru = define_gru_lstm_layers(num_layers, method, bidir_hidden_units, bidirectional, hnn_maxpooling, 
                                     selfattention, maxpooling, alignment)
    
        
    #Define size of Input layer for Siamese
    print(max_len)
    print(len_chars)
    input_word1 = Input(shape=(max_len, len_chars))
    input_word2 = Input(shape=(max_len, len_chars))

    #Set Sentence-Encoder Model 
    mask = Masking(mask_value=0, input_shape=(max_len, len_chars))(input_word1)
    print(mask)
    
    if hnn_maxpooling:
        final_input = set_gru_lstm_maxpooling_encoder(hnn_num_layers, gru, mask)
    else:
        final_input = set_gru_lstm_encoder(bidir_num_layers, gru, mask, selfattention, attention_method, maxpooling, shortcut)
    print(final_input)
    
    #Create Sentence-Encoder Model 
    SentenceEncoder = Model(input_word1, final_input) # take input and output
    print(type(SentenceEncoder))
    print(SentenceEncoder.summary())

    #context vector
    word1_representation = SentenceEncoder(input_word1)
    word2_representation = SentenceEncoder(input_word2)
    print(type(word1_representation))
    print(word1_representation)
    print(word2_representation)
    time_gru, unit_gru = get_tensor_shape(word1_representation)

    if alignment:
        #word1_representation = K.expand_dims(word1_representation) # add 1 dimension
        #print(word1_representation)
        #word2_representation = K.expand_dims(word2_representation) # add 1 dimension
        
        
        
        #Self Attention
        #att1 = SelfAttLayer()(word1_representation)
        #print(att1)
        #time_gru, unit_gru = get_tensor_shape(att1)
        
        #Remove None
        '''
        word1_representation_no_none = remove_none_to_tensor_shape(word1_representation)
        print(word1_representation_no_none)
        word2_representation_no_none = remove_none_to_tensor_shape(word2_representation)
        print(word2_representation_no_none)
        time_gru, unit_gru = get_tensor_shape(word2_representation_no_none)
        att1 = AlignmentAttentionLayer()([word1_representation_no_none, word2_representation_no_none])
        print(att1)
        att2 = AlignmentAttentionLayer()([word2_representation_no_none, word1_representation_no_none])
        print(att2)
        
        #Add None back
        att1 = add_none_to_tensor_shape(att1)
        print(att1)
        att2 = add_none_to_tensor_shape(att2)
        print(att2)
        '''
        
        
        att1 = AlignmentAttentionLayer()([word1_representation, word2_representation])
        att2 = AlignmentAttentionLayer()([word2_representation, word1_representation])
        print(type(att1))
        print(att1)
        time_gru, unit_gru = get_tensor_shape(att1)
        
        '''
        att1 = CustomDense(1)(K.transpose(att1))
        att2 = CustomDense(1)(K.transpose(att2))
        print(att1)
        time_gru, unit_gru = get_tensor_shape(att1)
        '''
        
        '''
        att1 = GlobalMaxPooling1DMasked(name = 'maxpooling'+'att1')(att1)
        print(att1)
        att2 = GlobalMaxPooling1DMasked(name = 'maxpooling'+'att2')(att2)
        print(att2)
        '''

        concat = concatenate([att1,att2])
        mul = multiply([att1,att2])
        sub = subtract([att1, att2])
    else:
        concat = concatenate([word1_representation, word2_representation])
        mul = multiply([word1_representation, word2_representation])
        sub = subtract([word1_representation, word2_representation])

    if onlyconcat:
        final_merge = concat
    else:
        final_merge = concatenate([concat, mul, sub])
    input_val = Dropout(0.01)(final_merge)
    print(input_val)
    #if alignment:
    #    input_val = Flatten()(input_val)
    #print(input_val)
    
    #Set Siamese Model with 2-inputs
    if hnn_maxpooling:
        dense_output = set_deep_network_maxpooling_layers(dnn_num_layers, dnn_hidden_units, input_val)
    else:
        dense_output = set_deep_network(dnn_num_layers, dnn_hidden_units, input_val)
    print(dense_output)
        
    
    #Create Siamese Model with 2-inputs and 1 output
    final_model = Model([input_word1, input_word2], dense_output) 
    final_model.summary()
    print('Compiling...')
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('Fitting...')
    #num_epochs = 2
    #3D it knows you have a 2D object, it loops over the 1st dimension
    #callbacks_list=[early_stop]
    callbacks_list=[checkpointer, early_stop]
    history = final_model.fit([train_data_1, train_data_2], train_labels, verbose = 0, validation_data=([test_data_1, test_data_2], test_labels), 
		    callbacks=callbacks_list, epochs=num_epochs)

    print('Predicting...')
    start_time = time.time()
    aux1 = final_model.predict([test_data_1, test_data_2], verbose = 0) #array of float
    #define match/unmatch
    aux = (aux1 > 0.5).astype('int32').ravel() #array of int, get 1 for a match
    return aux, (time.time() - start_time)


