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

def std_logit(h, W, std):
    main_dtype=h.dtype
    std=tf.saturate_cast(std, tf.float32)
    std=tf.cast(tf.less_equal(-4.0,std) & tf.less_equal(std,3.0), tf.float32)*std+tf.cast(tf.less(3.0,std), tf.float32)*(4.0-tf.exp(3.0-std))+tf.cast(tf.less(std, -4.0), tf.float32)*(-5+tf.exp(4.0+std))
    std=tf.saturate_cast(tf.exp(std), h.dtype)
    a=(4*(std**(-1)*(tf.exp(std)-1)-1))**(-1)
    h_norm=tf.norm(h, axis=-1)+1e-7
    h_norm=tf.expand_dims(h_norm, -1)
    W_norm=tf.norm(W, axis=0)+1e-7
    inner_product=tf.tensordot(h, W, axes=[[-1],[0]])########
    #return inner_product+tf.reduce_mean(std) #work
    cos=inner_product/h_norm/W_norm
    #return cos+tf.reduce_mean(std) #work
    PN=tf.cast(cos>1e-7, main_dtype)
    cos_abs=tf.abs(cos)+1e-7 #change from 1e-9->1e-7 because of fp16, 1e-7work, but 1e-9 not work
    ratio_pos=(a*tf.exp(std*cos_abs)-a)/cos_abs
    #return ratio_pos+tf.reduce_mean(std) #work
    return (2*PN*ratio_pos+(1-PN)*(ratio_pos*0+1))*inner_product
    #return inner_product
  
def bas_top(top_features, bayes_component, top_dimension, dtype):
    feature_shape_tensor=tf.shape(top_features)
    feature_shape=top_features.shape
    ndims=len(feature_shape)
    W=tf.get_variable(name='bas/W', shape=[feature_shape[-1] , top_dimension*bayes_component], dtype=dtype)
    std=tf.get_variable(name='bas/std', initializer=tf.zeros([top_dimension*bayes_component], dtype=dtype))
    logits=std_logit(top_features, W, std)
    if dtype==tf.float16:
        logits=tf.saturate_cast(logits, tf.float32)
    probs=advanced_softmax(logits)
    advanced_add_to_collections('bas_collection', probs, 'sense_probs')
    #probs=tf.reduce_sum(tf.concat([tf.expand_dims(x, -1) for x in tf.split(probs, bayes_component, -1)], -1), -1)
    probs=tf.reshape(probs, [-1, top_dimension*bayes_component])
    sense_initial_allocate=[]
    #print(top_dimension, bayes_component)
    for i in range(bayes_component):
        sense_initial_allocate.extend(list(range(top_dimension)))
    sense_initial_allocate=np.array(sense_initial_allocate).astype(np.int64)
    sense_allocate=tf.get_variable(name='bas/sa', initializer=sense_initial_allocate, trainable=False, dtype=tf.int64)
    advanced_add_to_collections('bas_collection', sense_allocate,  'sense_allocate')
    sense_allocate_ed=tf.expand_dims(sense_allocate, -1)
    sense_num=tf.constant(np.array(list(range(top_dimension*bayes_component))).astype(np.int64))
    sense_num_ed=tf.expand_dims(sense_num, -1)
    sense_values=tf.ones(shape=[top_dimension*bayes_component], dtype=tf.float32)
    sense_allocate_matrix=tf.SparseTensor(indices=tf.concat([sense_allocate_ed, sense_num_ed], axis=-1), values=sense_values, dense_shape=[top_dimension, top_dimension*bayes_component])
    advanced_add_to_collections('bas_collection', sense_allocate_matrix,  'sense_allocate_matrix')
    tf.transpose(tf.sparse.matmul(sense_allocate_matrix, tf.transpose(probs)))###
    probs=tf.transpose(tf.sparse.matmul(sense_allocate_matrix, tf.transpose(probs)))
    if ndims==3:
        probs=tf.reshape(probs,[feature_shape_tensor[0], feature_shape_tensor[1], top_dimension])
    logits=tf.log(probs)
    
    if dtype==tf.float16:
        logits=tf.saturate_cast(logits, tf.float16)
        probs=tf.saturate_cast(probs, tf.float16)
        advanced_add_to_collections('bas_collection', probs, 'word_probs')
    else:
        advanced_add_to_collections('bas_collection', probs, 'word_probs')
        pass
    
    usage=tf.get_variable(name='bas/usage', initializer=tf.zeros([top_dimension*bayes_component], dtype=tf.float32), trainable=False)
    advanced_add_to_collections('bas_collection', usage, 'usage')
    efficiency=tf.get_variable(name='bas/efficiency', initializer=tf.zeros([top_dimension], dtype=tf.float32), trainable=False)
    advanced_add_to_collections('bas_collection', efficiency, 'efficiency')
    word_count=tf.get_variable(name='bas/word_count', initializer=tf.zeros([top_dimension], dtype=tf.int32), trainable=False, dtype=tf.int32)
    advanced_add_to_collections('bas_collection', word_count, 'word_count')
    return logits
    
    
if __name__=='__main__':
    A=tf.get_variable(name='A', shape=[3,4,5], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.1))
    B=bas_top(A, 2, 10, tf.float32)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.reduce_sum(tf.exp(B), axis=-1)))