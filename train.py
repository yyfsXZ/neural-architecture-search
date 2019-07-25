import numpy as np
import csv
import os
import json

import random
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn_new as model_fn

import data_helper as dh

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 3  # number of layers of the state space
MAX_TRIALS = 100  # maximum number of models generated, adjust by xtpan from 250 to 2

MAX_EPOCHS = 3  # maximum number of epochs to train, adjust by xtpan from 10 to 2
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = False  # restore controller to continue training

MAX_SEQ_LENGTH = 30

MODEL_NAME = "textcnn"

# init data_helper
my_dh = dh.MyHelper(MAX_SEQ_LENGTH)
my_dh.initialize()
x_train, y_train, x_test, y_test = my_dh.read_input("../data/all_data.txt")

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='embedding', values=[100, 200, 300])
state_space.add_state(name='bidirection_lstm', values=[64, 128, 256])
#state_space.add_state(name='filters', values=[32, 64, 128, 256])	# Mi
state_space.add_state(name='filters', values=[16, 32, 64])	# Fawcar
state_space.add_state(name='kernel_height', values=[2, 3, 4, 5])
state_space.add_state(name='pool_weight', values=[2, 3, 4, 5])
#state_space.add_state(name='fc_size', values=[256, 512, 1024, 2048])	# Mi
state_space.add_state(name='fc_size', values=[256, 512])    # Fawcar
state_space.add_state(name="vocab_size", values=[my_dh.get_vocab_size()])
state_space.add_state(name="max_seq_length", values=[MAX_SEQ_LENGTH])
state_space.add_state(name="label_num", values=[len(my_dh.label2id.keys())])
# define model type; lstm / bilstm / lstm+bilstm / lenet
state_space.add_state(name="model_type", values=[MODEL_NAME])   

# print the state space being searched
state_space.print_state_space()

# laod train&test dataset
x_train = np.asarray(x_train, dtype=np.int32)
y_train = np.asarray(y_train, dtype=np.int32)
x_test = np.asarray(x_test, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

y_train = np.reshape(y_train, newshape=[y_train.shape[0], 1])
y_train = to_categorical(y_train, num_classes=my_dh.get_label_size())
y_test = np.reshape(y_test, newshape=[y_test.shape[0], 1])
y_test = to_categorical(y_test, num_classes=my_dh.get_label_size())

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# init saved_model path
pb_dir = "./model_pb"
os.system("rm -rf %s" % pb_dir)
os.mkdir(pb_dir)

# model name
os.mkdir("%s/%s" % (pb_dir, MODEL_NAME))

# save model.conf
wp = open("%s/model.conf" % pb_dir, 'w', encoding="utf-8")
model_conf = {"labels": "labels.dic", "epoch": 0, "max_sent_len": MAX_SEQ_LENGTH, "token_idx": "word.idx", "dnn_model": MODEL_NAME}
wp.write(json.dumps(model_conf))
wp.close()

# save labels.dic
sorted_labels = sorted(my_dh.label2id.items(), key=lambda x:x[1], reverse=False)
wp = open("%s/labels.dic" % pb_dir, 'w', encoding="utf-8")
for elem in sorted_labels:
    wp.write("%s\n" % elem[0])
wp.close()

# save word.idx
wp = open("%s/word.idx" % pb_dir, 'w', encoding="utf-8")
for idx in range(my_dh.get_vocab_size()):
    wp.write("%s\t%d\n" % (my_dh.get_char_by_id(idx), idx))
wp.close()
print("vocab size: %d" % my_dh.get_vocab_size())
#import sys
#sys.exit()
'''
# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(y_train.shape)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
'''

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA, model_dir="%s/%s" % (pb_dir, MODEL_NAME))

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()

# train for number of trails
for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  # get an action for the previous state

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

print("Total Reward : ", total_reward)
