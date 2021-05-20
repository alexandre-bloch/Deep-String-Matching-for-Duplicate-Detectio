#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
os.environ['KERAS_BACKEND'] = 'theano'
import sys
import csv
import time
import unicodedata
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Masking, Dense, Input, Dropout, LSTM, GRU, Bidirectional, MaxPooling1D, GlobalMaxPooling1D, Layer, Masking, Lambda, Permute, TimeDistributed  
#from keras.layers import Highway 
from tensorflow.keras.layers import concatenate, Reshape, Flatten, Activation, RepeatVector, Multiply
from tensorflow.keras.layers import Dot
#from tensorflow.keras.layers import dot

from tensorflow.keras import backend as K
#from tensorflow.keras import regularizers, constraints

import importlib

def set_keras_backend(backend):
    K_backend = K.backend()
    print(K_backend)
    if K_backend != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend  
    print(K_backend)


def func_build_tf_3D(zdir, tf_obj):
    tf_obj = tf.expand_dims(tf_obj, axis=0)
    print(tf_obj)
    
    '''
    for z in range(0, zdir):
        row = []
        row.append(count)
        matrix.append(row)
    '''
    
    return tf_obj
        
# Numpy reports the shape of 3D arrays in the order layers, rows, columns.        
def func_build_3D(zdir, matrix):
    matrix = np.expand_dims(matrix, axis=0)
    print(matrix)
    print(matrix.shape)
    
    '''
    for z in range(0, zdir):
        matrix[z,:,:] = 1
    print(matrix)
    '''
        
    return matrix
        

def func_build_matrix(rows, cols):
    #matrix = []

    #for r in range(0, rows):
    #    matrix.append([c for c in range(0, cols)])
    
    np.random.seed(8)    
    matrix = np.random.rand(rows, cols)
    
    '''
    count = 0.0 
    for r in range(0, rows):
        row = []
        for c in range(0, cols):
            count += 0.01
            row.append(count)
        matrix.append(row)
    '''

    return matrix

def func_divide_by_zero(x,y):
    try:
        return x/y
    except ZeroDivisionError as e:
        print(e)
        return 0



def get_diagonal_tf(x, kernel):
    
    temp_mut = tf.multiply(x, kernel)
    print(temp_mut)
    a_diag = tf.reduce_sum(temp_mut, 2, keepdims=True )
    print(a_diag)
    
    #tensor_dot = tf.tensordot(x, kernel, axes=[[1],[1]])
    #print(tensor_dot)
    #a_diag = tf.diag_part(tensor_dot)
    #print(a_diag)
    
    return a_diag


def dot_product_tf(x, kernel, axis1=1, axis2=1):
    """
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    print(x)
    print(kernel)
    
    temp_shape = K.int_shape(x)
    print(temp_shape)
    tuple_size = len(temp_shape)
    print(tuple_size)
    
    temp_shape_ker = K.int_shape(kernel)
    tuple_size_ker = len(temp_shape_ker)
    print(tuple_size_ker)
    
    #a_diag = K.batch_dot(x, kernel, axes=2) 
    #print(a_diag)
    
    if tuple_size == 2:
        dot_product = Dot(axes=axis1)([x, kernel]) #5*5
        print(dot_product)
    elif tuple_size == 3:
        if tuple_size_ker == 2:
            kernel_dim = K.expand_dims(kernel,0) #add 1 dim to the left
        dot_product = Dot(axes=(axis1,axis2))([x, kernel_dim]) #5*5
        print(dot_product)
        #dot_product1 = Reshape((1,))(dot_product1)
        #print(dot_product1)
    else:
        dot_product = -1
    result = dot_product
    
    return result

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    print(x)
    print(kernel)
    
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        #expand_dim = K.expand_dims(kernel) #add 1 dim to the right
        #expand_dim = K.expand_dims(kernel,0) #add 1 dim to the left
        expand_dim = kernel
        print(expand_dim)
        dot_pro = K.dot(x, expand_dim)
        #dot_pro = K.batch_dot(x, expand_dim, axes=[2,1])
        print(dot_pro)
        result = dot_pro
        #result = K.squeeze(dot_pro, axis=-1) #remove last dim
        print(result)
        return result
    else:
        return K.dot(x, kernel)
    
    

def get_tensor_shape(tensor_obj):
    #H, W, n_ch = tensor.shape.as_list()[1:]
    #print(K.shape(tensor_obj))
    temp_shape = K.int_shape(tensor_obj)
    #print(type(temp_shape))
    print(temp_shape)
    tuple_size = len(temp_shape)
    #print(tuple_size)
    
    time_gru = 0; unit_gru = 0; thir_gru = 0
    if temp_shape[0] is None:
        if tuple_size == 3:
            time_gru = temp_shape[1]
            unit_gru = temp_shape[2]
        elif tuple_size == 2:
            unit_gru = temp_shape[1]
    else:
        if tuple_size == 1:
            time_gru = temp_shape[0]
        if tuple_size == 2:
            time_gru = temp_shape[0]
            unit_gru = temp_shape[1]
        if tuple_size == 3:
            time_gru = temp_shape[1]
            unit_gru = temp_shape[2]
    #print(time_gru)
    #print(unit_gru)
    
    return time_gru, unit_gru 


def add_none_to_tensor_shape(tensor_obj):
    print(type(tensor_obj))
    print(tensor_obj)
    time_gru, unit_gru = get_tensor_shape(tensor_obj)
    temp_shape0 = tensor_obj.shape
    print(temp_shape0)
    temp_shape = K.int_shape(tensor_obj)
    print(temp_shape)
    tuple_size = len(temp_shape)
    
   
    d_type = 'float32' #'int32'
    #tensor_obj = Input(shape=((None,) + temp_shape0), dtype = 'int32')
    tensor_obj = Input(shape=(temp_shape0), dtype = d_type)
    tensor_obj.shape
    print(type(tensor_obj))
    print(tensor_obj)
    time_gru, unit_gru = get_tensor_shape(tensor_obj)
    
    '''
    if tuple_size == 1:
        tensor_obj = tf.reshape(tensor_obj, [-1] + [temp_shape[0]])
    elif tuple_size == 2:
        tensor_obj = tf.reshape(tensor_obj, [-1] + [temp_shape[0],temp_shape[1]])
    print(tensor_obj)
    time_gru, unit_gru = get_tensor_shape(tensor_obj)
    '''
    
    return tensor_obj


def remove_none_to_tensor_shape(tensor_obj):
    time_gru, unit_gru = get_tensor_shape(tensor_obj)
    print(time_gru)
    print(unit_gru)
    temp_shape = tensor_obj.shape
    temp_shape = K.int_shape(tensor_obj)
    print(temp_shape)
    tuple_size = len(temp_shape)
    print(tuple_size)
    
    #tensor_obj = tf.shape(tensor_obj)[0]
    #tensor_obj = tf.reshape(tensor_obj, shape=[tf.shape(tensor_obj)[0], temp_shape[0],temp_shape[1]])
    print(tensor_obj)
    time_gru, unit_gru = get_tensor_shape(tensor_obj)
    
    if temp_shape[0] is None:
        if tuple_size == 2:
            tensor_obj = tf.reshape(tensor_obj, [temp_shape[1]])
        elif tuple_size == 3:
            tensor_obj = tf.reshape(tensor_obj, [temp_shape[1],temp_shape[2]])
    print(tensor_obj)
    time_gru, unit_gru = get_tensor_shape(tensor_obj)
    
    return tensor_obj


    
def softmax(x, axis=-1):
    print(x)
    time_gru, unit_gru = get_tensor_shape(x)
    ndim = K.ndim(x)
    print(ndim)
    if ndim == 2: 
        result = K.softmax(x)
        print(result)
        return result
    elif ndim > 2:
        #temp_max = K.max(x, axis=axis, keepdims=True)
        e = K.exp(x)
        print(e)
        sum_weights = K.sum(e, axis=axis, keepdims=True)
        #sum_weights = tf.reduce_sum(ai, axis=1, keepdims=True)
        s = K.cast(sum_weights, K.floatx())
        print(s)
        result = e / s
        print(result)
        return result
    else: raise ValueError('Cannot apply softmax to a tensor that is 1D')
    

def flatten_tensor(ts):
    ts = Reshape(ts, (1,-1))
    ts = K.squeeze(ts)
    return ts


