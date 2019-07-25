import numpy as np
import os

from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, epochs=5, child_batchsize=128, acc_beta=0.8, clip_rewards=0.0, model_dir="model_pb"):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: float - to clip rewards in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

        self.model_dir = model_dir

    def get_rewards(self, model_fn, actions):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given actions
        '''
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)

            # generate a submodel given predicted actions
            model = model_fn(actions)  # type: Model
            model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

            # unpack the dataset
            X_train, y_train, X_val, y_val = self.dataset

            # train the model using Keras methods
            model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs,
                      verbose=1, validation_data=(X_val, y_val),
                      callbacks=[ModelCheckpoint('weights/temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=False)])

            # load best performance epoch in this training session
            model.load_weights('weights/temp_network.h5')

            # evaluate the model
            loss, acc = model.evaluate(X_val, y_val, batch_size=self.batchsize)

            # compute the reward
            reward = (acc - self.moving_acc)

            # if rewards are clipped, clip them in the range -0.05 to 0.05
            if self.clip_rewards:
                reward = np.clip(reward, -0.05, 0.05)

            # update moving accuracy with bias correction for 1st update
            if self.beta > 0.0 and self.beta < 1.0:
                self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
                self.moving_acc = self.moving_acc / (1 - self.beta_bias)
                self.beta_bias = 0

                reward = np.clip(reward, -0.1, 0.1)

            print()
            print("Manager: EWA Accuracy = ", self.moving_acc)

            # save model as pb
            print("Model input.name={}".format(model.input.op.name))
            print("Model output.name={}".format(model.output.op.name))
            h5_model = load_model("weights/temp_network.h5")
            frozen_graph = self.freeze_session(h5_model, K.get_session(), output_names=["y_prob/Softmax"])
            tf.train.write_graph(frozen_graph, self.model_dir, "saved_model.pb", as_text=False)

        # clean up resources and GPU memory
        network_sess.close()

        return reward, acc

    def freeze_session(self, model, session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        :param session: 需要转换的tensorflow的session
        :param keep_var_names:需要保留的variable，默认全部转换constant
        :param output_names:output的名字
        :param clear_devices:是否移除设备指令以获得更好的可移植性
        :return:
        """
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            # 如果指定了output名字，则复制一个新的Tensor，并且以指定的名字命名
            if len(output_names) > 0:
                for i in range(len(output_names)):
                    # 当前graph中复制一个新的Tensor，指定名字
                    tf.identity(model.outputs[i], name=output_names[i])
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph
