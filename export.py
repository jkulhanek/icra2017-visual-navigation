#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import sys
import os

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread
from scene_loader import THORDiscreteEnvironment as Environment

from utils.ops import sample_action

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import NUM_EVAL_EPISODES
from constants import VERBOSE

from constants import TASK_TYPE
from constants import TASK_LIST
import pickle

if __name__ == '__main__':

  device = "/cpu:0" # use CPU for display tool
  network_scope = TASK_TYPE
  list_of_tasks = TASK_LIST
  scene_scopes = list_of_tasks.keys()

  global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
                                        device=device,
                                        network_scope=network_scope,
                                        scene_scopes=scene_scopes)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
  checkpoint.model_checkpoint_path = os.path.normpath(os.path.join(__file__,'..\\checkpoints', 'checkpoint-10000085'))

  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")

  print('Exporting waights to pytorch')
  model = {}
  model['navigation'] = {
      'fc_siemense.weight': np.transpose(sess.run(global_network.W_fc1['navigation'])),
      'fc_siemense.bias': sess.run(global_network.b_fc1['navigation']),
      'fc_merge.weight': np.transpose(sess.run(global_network.W_fc2['navigation'])),
      'fc_merge.bias': sess.run(global_network.b_fc2['navigation']),
  }

  for key in global_network.W_fc3.keys():
    model[key] = {
        'fc1.weight': np.transpose(sess.run(global_network.W_fc3[key])),
        'fc1.bias': sess.run(global_network.b_fc3[key]),
        'fc2_policy.weight': np.transpose(sess.run(global_network.W_policy[key])),
        'fc2_policy.bias': sess.run(global_network.b_policy[key]),
        'fc2_value.weight': np.transpose(sess.run(global_network.W_value[key])),
        'fc2_value.bias': sess.run(global_network.b_value[key]),
    }
  
  print('Weights exported')
  print('Saving')
  pickle.dump(model, open( "weights.p", "wb" ))