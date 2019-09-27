""" This code is modified from the following paper.
Learnable mOdUle for Pooling fEatures (LOUPE)
Contains a collection of models (NetVLAD, NetRVLAD, NetFV and Soft-DBoW)
which enables pooling of a list of features into a single compact 
representation.

Reference:

Learnable pooling method with Context Gating for video classification
Antoine Miech, Ivan Laptev, Josef Sivic

"""
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from keras import initializers, layers
import keras.backend as K
import sys


# Keras version

class ContextGating(layers.Layer):
    """Creates a NetVLAD class.
    """
    def __init__(self, **kwargs):
        
        super(ContextGating, self).__init__(**kwargs)

    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.gating_weights = self.add_weight(name='kernel_W1',
                                      shape=(input_shape[-1], input_shape[-1]),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(input_shape[-1])),
                                      trainable=True)
        self.gating_biases = self.add_weight(name='kernel_B1',
                                      shape=(input_shape[-1],),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(input_shape[-1])),
                                      trainable=True)
        
        super(ContextGating, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        """
        In Keras, there are two way to do matrix multiplication (dot product)
        1) K.dot : AxB -> when A has batchsize and B doesn't, use K.dot
        2) tf.matmul: AxB -> when A and B both have batchsize, use tf.matmul
        
        Error example: Use tf.matmul when A has batchsize (3 dim) and B doesn't (2 dim)
        ValueError: Shape must be rank 2 but is rank 3 for 'net_vlad_1/MatMul' (op: 'MatMul') with input shapes: [?,21,64], [64,3]
        
        tf.matmul might still work when the dim of A is (?,64), but this is too confusing.
        Just follow the above rules.
        """
        gates = K.dot(inputs, self.gating_weights)
        gates += self.gating_biases
        gates = tf.sigmoid(gates)

        activation = tf.multiply(inputs,gates)
        return activation

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)



class NetVLAD(layers.Layer):
    """Creates a NetVLAD class.
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        super(NetVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.cluster_weights = self.add_weight(name='kernel_W1',
                                      shape=(self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_biases = self.add_weight(name='kernel_B1',
                                      shape=(self.cluster_size,),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_weights2 = self.add_weight(name='kernel_W2',
                                      shape=(1,self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.hidden1_weights = self.add_weight(name='kernel_H1',
                                      shape=(self.cluster_size*self.feature_size, self.output_dim),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)),
                                      trainable=True)
        
        super(NetVLAD, self).build(input_shape)  # Be sure to call this at the end

    def call(self, reshaped_input):
        """Forward pass of a NetVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        """
        In Keras, there are two way to do matrix multiplication (dot product)
        1) K.dot : AxB -> when A has batchsize and B doesn't, use K.dot
        2) tf.matmul: AxB -> when A and B both have batchsize, use tf.matmul
        
        Error example: Use tf.matmul when A has batchsize (3 dim) and B doesn't (2 dim)
        ValueError: Shape must be rank 2 but is rank 3 for 'net_vlad_1/MatMul' (op: 'MatMul') with input shapes: [?,21,64], [64,3]
        
        tf.matmul might still work when the dim of A is (?,64), but this is too confusing.
        Just follow the above rules.
        """
        activation = K.dot(reshaped_input, self.cluster_weights)
        
        activation += self.cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)
        
        a = tf.multiply(a_sum,self.cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,
            self.max_samples, self.feature_size])

        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = tf.reshape(vlad,[-1, self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = K.dot(vlad, self.hidden1_weights)

        return vlad

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_dim])




class NetRVLAD(layers.Layer):
    """Creates a NetRVLAD class (Residual-less NetVLAD).
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        super(NetRVLAD, self).__init__(**kwargs)
    
    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.cluster_weights = self.add_weight(name='kernel_W1',
                                      shape=(self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_biases = self.add_weight(name='kernel_B1',
                                      shape=(self.cluster_size,),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.hidden1_weights = self.add_weight(name='kernel_H1',
                                      shape=(self.cluster_size*self.feature_size, self.output_dim),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)),
                                      trainable=True)
        
        super(NetRVLAD, self).build(input_shape)  # Be sure to call this at the end

    def call(self, reshaped_input):
        """Forward pass of a NetRVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        """
        In Keras, there are two way to do matrix multiplication (dot product)
        1) K.dot : AxB -> when A has batchsize and B doesn't, use K.dot
        2) tf.matmul: AxB -> when A and B both have batchsize, use tf.matmul
        
        Error example: Use tf.matmul when A has batchsize (3 dim) and B doesn't (2 dim)
        ValueError: Shape must be rank 2 but is rank 3 for 'net_vlad_1/MatMul' (op: 'MatMul') with input shapes: [?,21,64], [64,3]
        
        tf.matmul might still work when the dim of A is (?,64), but this is too confusing.
        Just follow the above rules.
        """
        activation = K.dot(reshaped_input, self.cluster_weights)
        
        activation += self.cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,
            self.max_samples, self.feature_size])

        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = tf.reshape(vlad,[-1, self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = K.dot(vlad, self.hidden1_weights)

        return vlad

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_dim])


class SoftDBoW(layers.Layer):
    """Creates a Soft Deep Bag-of-Features class.
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        super(SoftDBoW, self).__init__(**kwargs)

    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.cluster_weights = self.add_weight(name='kernel_W1',
                                      shape=(self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_biases = self.add_weight(name='kernel_B1',
                                      shape=(self.cluster_size,),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.hidden1_weights = self.add_weight(name='kernel_H1',
                                      shape=(self.cluster_size, self.output_dim),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)),
                                      trainable=True)
        super(SoftDBoW, self).build(input_shape)  # Be sure to call this at the end

    def call(self, reshaped_input):
        """Forward pass of a Soft-DBoW block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        """
        In Keras, there are two way to do matrix multiplication (dot product)
        1) K.dot : AxB -> when A has batchsize and B doesn't, use K.dot
        2) tf.matmul: AxB -> when A and B both have batchsize, use tf.matmul
        
        Error example: Use tf.matmul when A has batchsize (3 dim) and B doesn't (2 dim)
        ValueError: Shape must be rank 2 but is rank 3 for 'net_vlad_1/MatMul' (op: 'MatMul') with input shapes: [?,21,64], [64,3]
        
        tf.matmul might still work when the dim of A is (?,64), but this is too confusing.
        Just follow the above rules.
        """
        activation = K.dot(reshaped_input, self.cluster_weights)
        
        activation += self.cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        bow = tf.reduce_sum(activation,1)
        bow = tf.nn.l2_normalize(bow,1)
        bow = K.dot(bow, self.hidden1_weights)
        return bow

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_dim])



class NetFV(layers.Layer):
    """Creates a NetVLAD class.
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        super(NetFV, self).__init__(**kwargs)

    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.cluster_weights = self.add_weight(name='kernel_W1',
                                      shape=(self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.covar_weights = self.add_weight(name='kernel_C1',
                                      shape=(self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_biases = self.add_weight(name='kernel_B1',
                                      shape=(self.cluster_size,),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_weights2 = self.add_weight(name='kernel_W2',
                                      shape=(1,self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.hidden1_weights = self.add_weight(name='kernel_H1',
                                      shape=(2*self.cluster_size*self.feature_size, self.output_dim),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)),
                                      trainable=True)
        super(NetFV, self).build(input_shape)  # Be sure to call this at the end

    def call(self, reshaped_input):
        """Forward pass of a NetFV block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        """
        In Keras, there are two way to do matrix multiplication (dot product)
        1) K.dot : AxB -> when A has batchsize and B doesn't, use K.dot
        2) tf.matmul: AxB -> when A and B both have batchsize, use tf.matmul
        
        Error example: Use tf.matmul when A has batchsize (3 dim) and B doesn't (2 dim)
        ValueError: Shape must be rank 2 but is rank 3 for 'net_vlad_1/MatMul' (op: 'MatMul') with input shapes: [?,21,64], [64,3]
        
        tf.matmul might still work when the dim of A is (?,64), but this is too confusing.
        Just follow the above rules.
        """

        covar_weights = tf.square(self.covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights,eps)

        activation = K.dot(reshaped_input, self.cluster_weights)
        
        activation += self.cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)
        
        a = tf.multiply(a_sum,self.cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,
            self.max_samples, self.feature_size])

        fv1 = tf.matmul(activation,reshaped_input)
        fv1 = tf.transpose(fv1,perm=[0,2,1])

        # computing second order FV
        a2 = tf.multiply(a_sum,tf.square(self.cluster_weights2)) 

        b2 = tf.multiply(fv1,self.cluster_weights2) 
        fv2 = tf.matmul(activation,tf.square(reshaped_input)) 
     
        fv2 = tf.transpose(fv2,perm=[0,2,1])
        fv2 = tf.add_n([a2,fv2,tf.scalar_mul(-2,b2)])

        fv2 = tf.divide(fv2,tf.square(covar_weights))
        fv2 = tf.subtract(fv2,a_sum)

        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
      
        fv2 = tf.nn.l2_normalize(fv2,1)
        fv2 = tf.reshape(fv2,[-1,self.cluster_size*self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2,1)

        fv1 = tf.subtract(fv1,a)
        fv1 = tf.divide(fv1,covar_weights) 

        fv1 = tf.nn.l2_normalize(fv1,1)
        fv1 = tf.reshape(fv1,[-1,self.cluster_size*self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1,1)

        fv = tf.concat([fv1,fv2],1)

        fv = K.dot(fv, self.hidden1_weights)
        return fv

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_dim])

