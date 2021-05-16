#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
os.environ['KERAS_BACKEND'] = 'theano'
import os
import sys
import csv
import time
import unicodedata
import numpy as np
import tensorflow as tf
from functionsDSM import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Masking, Dense, Input, Dropout, LSTM, GRU, Bidirectional, MaxPooling1D, GlobalMaxPooling1D, Layer, Masking, Lambda, Permute, TimeDistributed  
#from keras.layers import Highway 
from tensorflow.keras.layers import concatenate, Reshape, Flatten, Activation, RepeatVector, Multiply
from tensorflow.keras.layers import Dot
#from tensorflow.keras.layers import dot
from tensorflow.keras import backend as K

#import theano
#from theano.tensor import _shared
#if K.backend() == 'theano':
from tensorflow.keras import initializers, regularizers, constraints
#else:
#    from keras import initializations

    
class GlobalMaxPooling1DMasked1(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMaxPooling1DMasked1, self).__init__(**kwargs)
    def build(self, input_shape): super(GlobalMaxPooling1DMasked1, self).build(input_shape)
    def call(self, x, mask=None): 
        return super(GlobalMaxPooling1DMasked1, self).call(x)
    
class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMaxPooling1DMasked, self).__init__(**kwargs)
    def build(self, input_shape): super(GlobalMaxPooling1DMasked, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def call(self, x, mask=None): 
        return super(GlobalMaxPooling1DMasked, self).call(x)
   # def __len__(self):
   #     return len(self.supports_masking())

class MaxPooling1DMasked(MaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaxPooling1DMasked, self).__init__(**kwargs)
    def build(self, input_shape): super(MaxPooling1DMasked, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        return None
    def call(self, x, mask=None): 
        return super(MaxPooling1DMasked, self).call(x)


#units = output    
class Dense(Layer):
    def __init__(self, units):
        self.units = units
        super(Dense, self).__init__()
         
    def build(self, input_shape): 
        assert len(input_shape) == 3
        
        self.W = self.add_weight(
        name = 'W',
        shape = (input_shape[-1], self.units),
        initializer = 'random_normal',
        trainable = True) 
        
        self.b = self.add_weight(
            name = 'b', shape = (self.units, ), initializer = 'zeros', 
            trainable = True)
        
    def call(self, inputs): 
        return tf.matmul(inputs, self.W) + self.b    
    
    
class CustomDense(Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), #last dim of shape
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        neuron = tf.matmul(inputs, self.w) + self.b
        result = K.squeeze(neuron, axis=-1) # squeeze the last element
        
        return result

    def get_config(self):
        return {"units": self.units}    


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
       
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.inp_dimensions = input_shape[-1] #Last value of inp
        self.seq_length = input_shape[-2]  #Last but 1 value

        self.W = self.add_weight((self.inp_dimensions,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.inp_dimensions,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)
        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

class Attention0(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(Attention0, self).__init__(**kwargs)
    
    '''
    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        #self.trainable_weights = [self.W]
        self._trainable_weights = [self.W]
        super(Attention0, self).build(input_shape)  # be sure you call this somewhere!
    '''    
      
    '''
    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal", trainable = True)
        self.b=self.add_weight(name="att_bias",shape=(self.attention_dim,),initializer="normal", trainable = True)
        self.u=self.add_weight(name="u_bias",shape=(self.attention_dim,1),initializer="normal", trainable = True)        
        super(Attention0, self).build(input_shape)
    '''

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


# x is input1
class SelfAttLayer(Layer):
    def __init__(self, attention_method = 'softmax', **kwargs):
        self.attention_method = attention_method
        self.attention = None
        #self.init = initializations.get('normal')
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(SelfAttLayer, self).__init__(**kwargs)
    
    
    def build(self, input_shape):
        assert len(input_shape) == 3 # input_shape[0] is None
        self.inp_dimensions = input_shape[-1] #Last value of input
        self.seq_length = input_shape[-2]  #Last but 1 value of input
        num_units = 1
        self.W = self.init((self.inp_dimensions, num_units))
        self.b = self.init((self.seq_length, num_units))
        #self.trainable_weights = [self.W]
        #self._trainable_weights = [self.W]
        super(SelfAttLayer, self).build(input_shape)
    
    
    
    def call(self, x, mask=None):
        result = compute_selfattention(x, self.W, self.inp_dimensions, self.attention_method, self.b)
        return result
    
    
    '''
    def call(self, x):
        # ‘W’ is the weight of the layer and ‘a’ is the attention weights
        # the inputs ‘x’ has shape (seq_length, inp_dimensions) 
        # the layer weights ‘w’ has shape (inp_dimensions, 1)
        # we compute x * W to get shape (seq_length, 1)
        # We add the bias ‘b’ and pass the output through any activation layer getting vales of size (seq_length, 1)
        # We take a softmax of these values, getting attention weights in the range [0,1]
        # We explicitly ‘squeeze’ the (seq_length, 1) attention vector into a 1D array of size (seq_length) 
        # before computing the softmax. 
        # ‘Expand’ back the attention weights from (seq_length) to (seq_length, 1)
        # We multiply each attention weight by the respective word and sum up 
        e = K.squeeze(K.tanh(K.dot(x,self.W)), axis=-1)
        #e = K.squeeze(K.tanh(K.dot(x,self.W)+self.b), axis=-1)
        a = K.softmax(e)
        a = K.expand_dims(a, axis=-1)
        output = x*a
        attention_adjusted_output =  K.sum(output, axis=1)
        return attention_adjusted_output
    '''
    
    def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[-1])
    def compute_output_shape(self, input_shape): return self.get_output_shape_for(input_shape)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention_method': self.attention_method,
        })
        return config
    
    
# Class Alignment Attention Layer    
class AlignmentAttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(AlignmentAttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(AlignmentAttentionLayer, self).build(input_shape)
        
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, inputs, mask=None):
        result = compute_alignment_attention(inputs, mask)
        return result
        
    def compute_output_shape(self, input_shape): return input_shape[0]
    
    
class AttentionLayer(Layer): 
    
    def __init__(self, **kwargs):    
        self.init = initializers.get('normal')
        super(AttentionLayer, self).__init__(**kwargs)
    
    '''  
    def build(self, input_shape):
        self.inp_dimensions = input_shape[-1] #Last value of inp
        self.W = self.init((self.inp_dimensions,))
        #self.trainable_weights = [self.W]
        self._trainable_weights = [self.W]
        super(AttentionLayer, self).build(input_shape)
    '''
    
       
    def build(self, input_shape):
        assert len(input_shape) == 3
        # Define the shape of the weights and bias in this layer with 1 single neuron
        
        self.inp_dimensions = input_shape[-1] #Last value of inp:256
        self.seq_length = input_shape[-2]  #Last but 1 value:19
        num_units = 1
        
        self.W = self.init((self.inp_dimensions, num_units))
        self.b = self.init((self.seq_length, num_units))
        #self.W = self.add_weight((self.inp_dimensions, num_units), initializer='normal')
        #self.b = self.add_weight((self.seq_length, num_units), initializer='zero')
        
        super(AttentionLayer, self).build(input_shape)
    
    
        
    '''   
    def call(self, x):
        # ‘W’ is the weight of the layer and ‘a’ is the attention weights
        # the inputs ‘x’ has shape (seq_length, inp_dimensions) 
        # the layer weights ‘w’ has shape (inp_dimensions, 1)
        # we compute x * W to get shape (seq_length, 1)
        # We add the bias ‘b’ and pass the output through any activation layer getting vales of size (seq_length, 1)
        # We take a softmax of these values, getting attention weights in the range [0,1]
        # We explicitly ‘squeeze’ the (seq_length, 1) attention vector into a 1D array of size (seq_length) 
        # before computing the softmax. 
        # ‘Expand’ back the attention weights from (seq_length) to (seq_length, 1)
        # We multiply each attention weight by the respective word and sum up 
        e = K.squeeze(K.tanh(K.dot(x,self.W)+self.b), axis=-1)
        a = K.softmax(e)
        a = K.expand_dims(a, axis=-1)
        output = x*a
        output =  K.sum(output, axis=1)
        return output
    '''
    
    # Functional approach
    def call(self, x):
        #e = K.tanh(K.dot(x,self.W))
        e = K.tanh(K.dot(x,self.W)+self.b)
        e = Flatten()(e)
        a = Activation('softmax')(e) #1D array of size (seq_length)
        
        # Don't manipulate 'a'. It needs to be returned intact
        temp = RepeatVector(self.inp_dimensions)(a) # (?,seq_length) becomes (?,inp_dimensions,seq_length)
        temp = Permute([2,1])(temp) # change from (?,inp_dimensions,seq_length) to (?,seq_length,inp_dimensions) like ‘x’
        output = Multiply()([x,temp]) # Apply weight to each of the inp_dimensions dim
        output = Lambda(lambda values: K.sum(values, axis=1))(output)
            
        #return a, output
        return output





def compute_proba(x, W, inp_dimensions=0, b=0):
    # computes a probability distribution over the timesteps
    # uses 'max trick' for numerical stability
    # reshape is done to avoid issue with Tensorflow
    # and 1-dimensional weights
        
    mask = None
    proba1 = True
        
    print(x)
    print(W)
        
    temp_shape = K.int_shape(x)
    print(temp_shape)
    tuple_size = len(temp_shape)
    print(tuple_size)
    
    if tuple_size == 2:
        dot_pro = K.dot(x,W)+b
    elif tuple_size == 3:    
        # Dot product on 3d shape
        axis1=2; axis2=1
        dot_pro = dot_product_tf(x, W, axis1, axis2)
    print(dot_pro)
        
    if proba1:
        e = dot_pro
    else:   
        #e = K.tanh(dot_product(x, K.transpose(W)))
        e = K.tanh(dot_pro) # takes values in [-1,1]  
    print(e)
    time_gru, unit_gru = get_tensor_shape(e)
    
    ai = K.exp(e)
    print(ai)
    time_gru, unit_gru = get_tensor_shape(ai)

    # masked timesteps have zero weight
    if mask is not None:
        mask = K.cast(mask, K.floatx())
        ai = ai * mask
    
    if tuple_size == 2:
        #sum_weights = K.sum(ai, axis=1, keepdims=True)
        sum_weights = tf.reduce_sum(ai, axis=0, keepdims=True)
    elif tuple_size == 3:
        sum_weights = tf.reduce_sum(ai, axis=1, keepdims=True)
    print(sum_weights)
    noise = K.epsilon()
    print(noise)
    
    if proba1:
        denom = sum_weights + noise
    else:
        denom = K.cast(sum_weights + noise, K.floatx())
    print(denom)
    a = ai/denom
    print(a)
    time_gru, unit_gru = get_tensor_shape(a)
    output = x*a # each weight is multiplied by the dimensions
    print(output)
    time_gru, unit_gru = get_tensor_shape(output)
    #result = K.sum(output, axis=1) # sum along the dimension
    result = tf.reduce_sum(output, axis=1, keepdims=False)
    print(result)
    time_gru, unit_gru = get_tensor_shape(result)
    
    '''
    if proba1:
        x_shape = K.shape(x)
        print(x_shape)
        x_shape_start = x_shape[0]
        print(x_shape_start)
        x_shape_end = x_shape[1]
        print(x_shape_end)
        #logits = K.reshape(dot_pro, (x_shape[0], x_shape[1]))
        max_trick = K.max(dot_pro, axis=-1, keepdims=True)
        print(max_trick)
        #ai = K.exp(dot_pro - max_trick)
        
        
        #att_weights = K.expand_dims(att_weights)
        #print(att_weights)
        
    else:
         
        #a = K.expand_dims(weights, axis=-1)
        #time_gru, unit_gru = get_tensor_shape(a)
        
    '''
    
    return result 


def compute_selfattention(x, W, inp_dimensions=0, attention_method = 'softmax', b=0):
    # ‘W’ is the weight of the layer and ‘a’ is the attention weights
    # the inputs ‘x’ has shape (seq_length, inp_dimensions) 
    # the layer weights ‘w’ has shape (inp_dimensions, 1)
    # we compute x * W to get shape (seq_length, 1)
    # We add the bias ‘b’ and pass the output through any activation layer getting vales of size (seq_length, 1)
    # We take a softmax of these values, getting attention weights in the range [0,1]
    # We explicitly ‘squeeze’ the (seq_length, 1) attention vector into a 1D array of size (seq_length) 
    # before computing the softmax. 
    # ‘Expand’ back the attention weights from (seq_length) to (seq_length, 1)
    # We multiply each attention weight by the respective word and sum up 
    
    print(type(x))
    time_gru_x, unit_gru_x = get_tensor_shape(x)
    time_gru, unit_gru = get_tensor_shape(W)
    print(inp_dimensions)
    
    #x = Dense(unit_gru_x, activation='sigmoid', name='denseOutput')(x)
    #time_gru_x, unit_gru_x = get_tensor_shape(x)
        
    if attention_method == 'softmax':
        functional = False
        
        if functional:
            dot_pro = K.dot(x,W)+b
            print(dot_pro)
            time_gru, unit_gru = get_tensor_shape(dot_pro)
            e = K.tanh(dot_pro)
            print(e)
            time_gru, unit_gru = get_tensor_shape(e)
            #e = Flatten()(e) # no
            #print(e)
            time_gru, unit_gru = get_tensor_shape(e)
            a = Activation('softmax')(e) #2D shape of size (seq_length)
            print(a)
            time_gru, unit_gru = get_tensor_shape(a)
            
            
            # Don't manipulate 'a'. It needs to be returned intact
            temp = RepeatVector(inp_dimensions)(a) # repeat (?,seq_length) inp_dimensions times so it becomes (?,inp_dimensions,seq_length)
            print(temp)
            time_gru, unit_gru = get_tensor_shape(temp)
            temp = Permute([2,1])(temp) # change from (?,inp_dimensions,seq_length) to (?,seq_length,inp_dimensions) like ‘x’
            print(temp)
            time_gru, unit_gru = get_tensor_shape(temp)
            output = Multiply()([x,temp]) # Apply weight to each of the inp_dimensions dim
            print(output)
            time_gru, unit_gru = get_tensor_shape(output)
            result = Lambda(lambda values: K.sum(values, axis=1))(output)
            print(result)
            time_gru, unit_gru = get_tensor_shape(result)
        else:
            dot_pro = K.dot(x,W)+b
            print(dot_pro)
            time_gru, unit_gru = get_tensor_shape(dot_pro)
            e = K.tanh(dot_pro)
            print(e)
            time_gru, unit_gru = get_tensor_shape(e)
            
            # Squeeze
            #e = K.squeeze(e, axis=-1) # squeeze the last element
            #print(e)
            #time_gru, unit_gru = get_tensor_shape(e)
            #a = K.softmax(e)
            #time_gru, unit_gru = get_tensor_shape(a)
            #a = K.expand_dims(a, axis=-1)
            #print(a)
            #time_gru, unit_gru = get_tensor_shape(a)
            
            # No Squeeze
            a = K.softmax(e, axis=0)
            print(a)
            
            output = x*a
            #output = Multiply()([x,a]) #elementwise multiplication
            print(output)
            time_gru, unit_gru = get_tensor_shape(output)
            result =  K.sum(output, axis=1) # sum along the dimension
            print(result)
            time_gru, unit_gru = get_tensor_shape(result)
        
    elif attention_method == 'proba':
        result = compute_proba(x, W)
    else:
        result = -1
    
    #return ai
    #weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
    #weighted_input = x*weights.dimshuffle(0,1,'x')
    #return weighted_input.sum(axis=1)
    
    return result
    
        
# Weight input1
def compute_alignment_attention(inputs, mask=None):   
    input1 = inputs[0]
    input2 = inputs[1]
    print(input1)

    # dot product
    axis1=2; axis2=2
    eij = dot_product_tf(input1, input2, axis1, axis2)
    #eij = dot_product(input1, K.transpose(input2))
    print(eij)
    eij = K.tanh(eij)
    print(eij)
    a = softmax(eij, axis=1)
    print(a)
    
    a = tf.reduce_sum(a, axis=2, keepdims=True)
    print(a)
    
    
    '''
    a = K.squeeze(a, axis=0) # squeeze the first element
    print(a)
    diag_a = tf.linalg.tensor_diag_part(a)
    print(diag_a) 
    a = K.expand_dims(diag_a) # add 1 dimension
    print(a)
    '''
    
    # Create 2D diagonal array
    '''
    a_diag = K.batch_dot(input1, input2, axes=1) 
    print(a_diag)
    #a.numpy()
    print(a)
    a_diag = np.diagonal(a, axis1 = 1, axis2 = 2)
    a = K.transpose(a_diag)
    print(a)
    print(a.shape)
    '''
    
    '''
    a_diag = get_diagonal_tf(input1, input2)
    print(a_diag)
    a_diag = K.tanh(a_diag)
    print(a_diag)
    sum_weights = tf.reduce_sum(a_diag, axis=1, keepdims=False)
    print(sum_weights)
    sum_value = tf.gather(sum_weights, 0)
    print(sum_value)
    noise = K.epsilon()
    denom = K.cast(sum_value + 10*noise, K.floatx())
    a = a_diag/denom
    print(a)
    '''
    
    output = input1*a
    print(output)
    result = tf.reduce_sum(output, axis=1, keepdims=False)
    print(result)
    time_gru, unit_gru = get_tensor_shape(result)
    
    return result    
    
# Weight input1
def compute_alignment_attention1(inputs, mask=None):   
    input1 = inputs[0]
    input2 = inputs[1]
    print(input1)

    eij = dot_product(input1, input2)
    #eij = dot_product(input1, K.transpose(input2))
    print(eij)
    eij = K.tanh(eij)
    print(eij)
    a = softmax(eij, axis=1)
    print(a)
    #a = K.expand_dims(a) # add 1 dimension
    #print(a)
    #weighted_input = input1*a
    weighted_input = dot_product(a,K.transpose(input1))

    result = weighted_input
    #result = K.sum(weighted_input, axis=1)
    print(result)
    return result

# Weight input1 (last hidden state) 
def compute_alignment_attention0(inputs, mask=None):
        input1 = inputs[0]
        input2 = inputs[1]
        print(input1)
        time_gru, unit_gru = get_tensor_shape(input1)
        print(input2)
        time_gru, unit_gru = get_tensor_shape(input2)

        # compute the score
        eij = K.dot(input1, K.transpose(input2))
        #eij = dot_product(input1, K.transpose(input2))
        print(eij)
        eij = K.tanh(eij)
        a = softmax(eij, axis=1)
        print(a)
        #a = K.expand_dims(a) # add 1 dimension
        #print(a)
        print(input1)
        
        #use dot product
        weighted_input = K.dot(a,input1)
        print(weighted_input)
        #result = K.squeeze(weighted_input, axis=-1) # squeeze the last element
        result = weighted_input
        print(result)
        
        # weight the input1 with the score
        '''
        weighted_input = input1*a
        print(weighted_input)
        result = K.sum(weighted_input, axis=1)
        print(result)
        '''
        
        return result






