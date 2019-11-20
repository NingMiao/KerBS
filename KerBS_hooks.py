#!/usr/bin/env python3
#Hooks for dynamically allocating senses.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime
import tensorflow as tf
from tensorflow import gfile
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util

import pickle as pkl
# try:
#     from horovod.tensorflow.mpi_ops import broadcast
# except:
#     pass
from misc import advanced_get_collection
from reallocate import get_new_sense_allocate

import horovod.tensorflow as hvd
from horovod.tensorflow.mpi_ops import broadcast
from horovod.tensorflow.mpi_ops import _allreduce as allreduce
from horovod.tensorflow.mpi_ops import allgather

class HvdReallocateHook(tf.train.SessionRunHook):
    """ Define the hook to display training loss, training speed and
    learning rate every n steps and determine when to stop. """

    def __init__(self,
                 reallocate_steps=100,
                 is_chief=True):
        """ Initializes the hook.

        Args:
            checkpoint_dir: A string, base directory for the checkpoint files.
            display_steps: A python integer, display every N steps.
            maximum_train_steps: A python integer, the maximum training steps.
            do_summary: Whether to save summaries when display.
            is_chief: Whether this is the chief process.do_summary:
        """

        tf.logging.info("Create HvdReallocateHook.")

        self._reallocate_steps = reallocate_steps
        self._is_chief = is_chief  

        ###
        self._collection_dict = {}
        name_list=['label_ids', 'mask', 'word_probs', 'sense_probs', 'sense_allocate', 'usage', 'efficiency', 'sense_allocate_matrix', 'word_count']
        for item in name_list:
            self._collection_dict[item] = advanced_get_collection('kerbs_collection', item)
        vocab_size=self._collection_dict['efficiency'].shape[-1]
        
        bayes_component=self._collection_dict['usage'].shape[-1] // vocab_size
        label_one_hot=tf.reshape(tf.one_hot(indices=self._collection_dict['label_ids'], depth=vocab_size), [-1, vocab_size])
        mask=tf.reshape(self._collection_dict['mask'], [-1])
        mask_expanded=tf.expand_dims(mask, -1)
        masked_label_one_hot=label_one_hot*mask_expanded
        step_efficiency=tf.reshape(self._collection_dict['word_probs'], [-1, vocab_size])*masked_label_one_hot
        self.step_efficiency=tf.reduce_sum(step_efficiency, axis=0)
        self.word_num=tf.cast(tf.reduce_sum(masked_label_one_hot, axis=0), tf.int32)
        sense_allocate_matrix=self._collection_dict['sense_allocate_matrix']
        sense_probs=self._collection_dict['sense_probs']
        label_onehot_reshape_transpose=tf.transpose(label_one_hot)
        label_sense_multi_hot=tf.transpose(tf.sparse.matmul(tf.sparse.transpose(sense_allocate_matrix), label_onehot_reshape_transpose))
        masked_label_sense_multi_hot=mask_expanded*label_sense_multi_hot
        self.step_usage=tf.reduce_sum(tf.reshape(sense_probs, [-1, vocab_size*bayes_component])*masked_label_sense_multi_hot, axis=0)
        def build_assign_op(input_tensor, accumulate_variable):
            op_add=tf.assign_add(accumulate_variable, input_tensor)
            op_zero=tf.assign(accumulate_variable, tf.zeros_like(accumulate_variable, dtype=accumulate_variable.dtype))
            op_allreduce=allreduce(accumulate_variable)
            return op_add, op_zero, op_allreduce
            
        self.word_num_update, self.word_num_zero, self.word_num_allreduce = build_assign_op(self.word_num, self._collection_dict['word_count'])
        self.efficiency_update, self.efficiency_zero, self.efficiency_allreduce = build_assign_op(self.step_efficiency, self._collection_dict['efficiency'])
        self.usage_update, self.usage_zero,  self.usage_allreduce= build_assign_op(self.step_usage, self._collection_dict['usage'])
        self.new_sense_allocate=tf.placeholder(shape=[vocab_size*bayes_component], dtype=tf.int64)
        sense_allocate_assign_op=tf.assign(self._collection_dict['sense_allocate'], self.new_sense_allocate)
        with tf.control_dependencies([sense_allocate_assign_op]):
            sense_allocate_broadcast_op=broadcast(self._collection_dict['sense_allocate'], 0)
        self.sense_allocate_update=tf.group(*[sense_allocate_assign_op, sense_allocate_broadcast_op])
        
        self._fetch_args={}
        global_step = training_util.get_global_step()
        self._fetch_args["global_step"] = global_step
        self._fetch_args['word_num']=self.word_num_update
        self._fetch_args['efficiency']=self.efficiency_update
        self._fetch_args['usage']=self.usage_update
        self._fetch_args['word_count']=self._collection_dict['word_count']
        
        self._allreduce_args={}
        self._allreduce_args['word_num']=self.word_num_allreduce
        self._allreduce_args['efficiency']=self.efficiency_allreduce
        self._allreduce_args['usage']=self.usage_allreduce
        self._allreduce_args['sense_allocate']=self._collection_dict['sense_allocate']
        
        self._zero_args={}
        self._zero_args['word_num']=self.word_num_zero
        self._zero_args['efficiency']=self.efficiency_zero
        self._zero_args['usage']=self.usage_zero
        
        
    def begin(self):
        """ Creates StepTimer and SummaryWriter. """
        self._timer = StepTimer(every_steps=self._reallocate_steps)
        pass
    
    def after_create_session(self, session, coord):
        pass
    
    def before_run(self, run_context):
        """ Run self._fetch_args
        """
        return tf.train.SessionRunArgs(self._fetch_args)

    def after_run(self, run_context, run_values):
        """ Get the updated sense allocation matrix and broadcast it.
        """
        global_step = run_values.results.pop("global_step")
        if self._timer.should_trigger_for_step(global_step-1):
            self._timer.update_last_triggered_step(global_step)
            allreduce_args=run_context.session.run(self._allreduce_args)
            #tf.logging.info(str(allreduce_args))
                    
            new_sense_allocate=get_new_sense_allocate(allreduce_args)
            run_context.session.run(self.sense_allocate_update, feed_dict={self.new_sense_allocate: new_sense_allocate})
            run_context.session.run(self._zero_args)
            
    

