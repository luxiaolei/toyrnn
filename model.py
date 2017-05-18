    
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer

class LSTMModel:
    
    def __init__(self, w_s, n_l, h_s, lr, l1_coe=0., l2_coe=0., clip=1):
        
        self.n_f = 5
        self.n_c = 2

        self.w_s = w_s
        self.n_l = n_l
        self.h_s = h_s
    

        self.l1_coe = l1_coe
        self.l2_coe = l2_coe
        self.clip = clip

        self.lr = lr
    
        self._define_model()

        self._define_lossNtrain()

    
    def _define_model(self):
        
        with tf.name_scope('input'):
            # `X_sliced` shape [b_s, w_s, n_f]
            # `Y_sliced` shape [b_s, ]
            self.x    = tf.placeholder(tf.float32, [None, self.w_s, self.n_f], name='X') 
            self.state_placeholder = tf.placeholder(tf.float32, [self.n_l, 2, None, self.h_s], 'state_placeholder') # None dim is b_s
            self.y = tf.placeholder(tf.int32, [None,], name='Yr_p') 
            self.drop_out = tf.placeholder(dtype=tf.float32,shape=[], name='drop_out')
            
        with tf.variable_scope("rnn",initializer=xavier_initializer(uniform=1)) as scope:
            layers = tf.unstack(self.state_placeholder, axis=0)
            rnn_tuple_state = tuple(
                    [tf.contrib.rnn.LSTMStateTuple(layer[0],layer[1])
                    for layer in layers])

            cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_s, 
                                                        activation=tf.nn.relu, 
                                                        layer_norm=True,#self.layer_norm, 
                                                        dropout_keep_prob=self.drop_out)
            for _ in range(self.n_l)]
                                                        
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            
            # `rnn_outputs` [b_s, w_s, h_s]
            # `final_state` is a `n_l` length tuple with each element
            # as a two sized tuple each has shape [b_s, h_s]
            rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell,self.x, parallel_iterations=60, initial_state=rnn_tuple_state)


        with tf.variable_scope('logits'):
            # Shape [b_s, num_outputs] 

            logits = tf.contrib.layers.fully_connected(rnn_outputs[:, -1, :], 
                                                    num_outputs = 2,  
                                                    activation_fn= None)

            self.pre_probs = tf.nn.softmax(logits)

            
    def _define_lossNtrain(self):
        with tf.name_scope('loss'):



            self.lossX = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pre_probs, 
                                                                labels=self.y)
            #self.loss = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pre_probs), reduction_indices=1))

            
            rnn_weights = [w for w in tf.trainable_variables() if 'fully_connected' not in w.name]
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self.l1_coe, scope=None)
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_coe, scope=None)
            
            l1_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, rnn_weights)
            l2_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, rnn_weights)
            
            self.loss = self.lossX + l1_penalty + l2_penalty


        with tf.name_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            opt = tf.train.AdamOptimizer(self.lr)
            gvs = opt.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -self.clip, self.clip), var) for grad, var in gvs]
            self.train_op = opt.apply_gradients(capped_gvs, global_step=self.global_step)
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
