#!/usr/bin/env python3
#Use kerbs_top to replace Softmax function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from misc import advanced_add_to_collections


def advanced_softmax(logits, mask=None):
    """ Computes softmax function manually.

    Avoids numeric overflow.

    Args:
        logits: A Tensor. The softmax will apply on the last dimension of it.
        mask: A Tensor with the same shape as `logits`.

    Returns: The softmax results.
    """
    num_shapes = logits.get_shape().ndims
    if mask is not None:
        scores_exp = tf.exp(
            logits - tf.reduce_max(
                logits, axis=num_shapes - 1, keepdims=True)) * mask
    else:
        scores_exp = tf.exp(
            logits - tf.reduce_max(
                logits, axis=num_shapes - 1, keepdims=True))
    scores_sum = tf.reduce_sum(scores_exp, axis=num_shapes - 1, keepdims=True)
    x_sm = scores_exp / scores_sum
    return x_sm

def theta_logit(h, W, theta):
    """ Variable kernel. Used to replace inner product.

    Args:
        h: A Tensor. Output vector.
        W: A Tensor. Sense or word embeddings
        theta: theta

    Returns: Output of kernel function.
    """
    main_dtype=h.dtype
    theta=tf.saturate_cast(theta, tf.float32)
    theta=tf.cast(tf.less_equal(-4.0,theta) & tf.less_equal(theta,3.0), tf.float32)*theta+tf.cast(tf.less(3.0,theta), tf.float32)*(4.0-tf.exp(3.0-theta))+tf.cast(tf.less(theta, -4.0), tf.float32)*(-5+tf.exp(4.0+theta))
    theta=tf.saturate_cast(tf.exp(theta), h.dtype)
    a=(4*(theta**(-1)*(tf.exp(theta)-1)-1))**(-1)
    h_norm=tf.norm(h, axis=-1)+1e-7
    h_norm=tf.expand_dims(h_norm, -1)
    W_norm=tf.norm(W, axis=0)+1e-7
    inner_product=tf.tensordot(h, W, axes=[[-1],[0]])
    cos=inner_product/h_norm/W_norm
    PN=tf.cast(cos>1e-7, main_dtype)
    cos_abs=tf.abs(cos)+1e-7 #change from 1e-9->1e-7 because of fp16, 1e-7work, but 1e-9 not work
    ratio_pos=(a*tf.exp(theta*cos_abs)-a)/cos_abs
    return (2*PN*ratio_pos+(1-PN)*(ratio_pos*0+1))*inner_product
  
def kerbs_top(top_features, bayes_component, top_dimension, dtype):
    """ KerBS top layer. Used to directly replace Softmax.

    Args:
        top_features: A Tensor. Output vector.
        bayes_component: A int number. Average sense number per word.
        top_dimension: A int number. Vocab size (not total sense size).
        dtype: dtype

    Returns: Logits of each word in vocab.
    """
    feature_shape_tensor=tf.shape(top_features)
    feature_shape=top_features.shape
    ndims=len(feature_shape)
    
    #Build sense embedding and theta
    W=tf.get_variable(name='kerbs/W', shape=[feature_shape[-1] , top_dimension*bayes_component], dtype=dtype)
    theta=tf.get_variable(name='kerbs/theta', initializer=tf.zeros([top_dimension*bayes_component], dtype=dtype))
    
    #Get sense logits
    sense_logits=theta_logit(top_features, W, theta)
    if dtype==tf.float16:
        sense_logits=tf.saturate_cast(sense_logits, tf.float32)
    
    #Get sense probs
    probs=advanced_softmax(sense_logits)
    advanced_add_to_collections('kerbs_collection', probs, 'sense_probs')
    probs=tf.reshape(probs, [-1, top_dimension*bayes_component])
    
    #Build a sparse matrix to control sense allocation
    sense_initial_allocate=[]
    for i in range(bayes_component):
        sense_initial_allocate.extend(list(range(top_dimension)))
    sense_initial_allocate=np.array(sense_initial_allocate).astype(np.int64)
    sense_allocate=tf.get_variable(name='kerbs/sa', initializer=sense_initial_allocate, trainable=False, dtype=tf.int64)
    advanced_add_to_collections('kerbs_collection', sense_allocate,  'sense_allocate')
    sense_allocate_ed=tf.expand_dims(sense_allocate, -1)
    sense_num=tf.constant(np.array(list(range(top_dimension*bayes_component))).astype(np.int64))
    sense_num_ed=tf.expand_dims(sense_num, -1)
    sense_values=tf.ones(shape=[top_dimension*bayes_component], dtype=tf.float32)
    sense_allocate_matrix=tf.SparseTensor(indices=tf.concat([sense_allocate_ed, sense_num_ed], axis=-1), values=sense_values, dense_shape=[top_dimension, top_dimension*bayes_component])
    advanced_add_to_collections('kerbs_collection', sense_allocate_matrix,  'sense_allocate_matrix')
    tf.transpose(tf.sparse.matmul(sense_allocate_matrix, tf.transpose(probs)))###
    probs=tf.transpose(tf.sparse.matmul(sense_allocate_matrix, tf.transpose(probs)))
    if ndims==3:
        probs=tf.reshape(probs,[feature_shape_tensor[0], feature_shape_tensor[1], top_dimension])
    logits=tf.log(probs)
    
    if dtype==tf.float16:
        logits=tf.saturate_cast(logits, tf.float16)
        probs=tf.saturate_cast(probs, tf.float16)
        advanced_add_to_collections('kerbs_collection', probs, 'word_probs')
    else:
        advanced_add_to_collections('kerbs_collection', probs, 'word_probs')
        pass
    
    #Build sense usage
    usage=tf.get_variable(name='kerbs/usage', initializer=tf.zeros([top_dimension*bayes_component], dtype=tf.float32), trainable=False)
    advanced_add_to_collections('kerbs_collection', usage, 'usage')
    
    #Build word log_P
    efficiency=tf.get_variable(name='kerbs/efficiency', initializer=tf.zeros([top_dimension], dtype=tf.float32), trainable=False)
    advanced_add_to_collections('kerbs_collection', efficiency, 'efficiency')
    
    #Build word count
    word_count=tf.get_variable(name='kerbs/word_count', initializer=tf.zeros([top_dimension], dtype=tf.int32), trainable=False, dtype=tf.int32)
    advanced_add_to_collections('kerbs_collection', word_count, 'word_count')
    return logits
    
    
if __name__=='__main__':
    A=tf.get_variable(name='A', shape=[3,4,5], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.1))
    B=kerbs_top(A, 2, 10, tf.float32)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.reduce_sum(tf.exp(B), axis=-1)))