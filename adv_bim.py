import os
import util
import model
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as itr
from scipy.misc import imread
from PIL import Image
from random import randrange
from collections import Counter

#cleverhans
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model, KerasModelWrapper
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, LBFGS, BasicIterativeMethod
from cleverhans.utils import AccuracyReport

#keras
import keras
from keras import __version__
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
print("Finished Import")

def main(argv):
    report = AccuracyReport()
    print("Start Main")
    tf.logging.set_verbosity(tf.logging.INFO)
    input_shape = (32,32,3)
    num_classes = 10
    x_train = np.load('/work/cse496dl/shared/hackathon/05/cifar10_train_data.npy')
    y_train = np.load('/work/cse496dl/shared/hackathon/05/cifar10_train_labels.npy')
    x_test = np.load('/work/cse496dl/shared/hackathon/05/cifar10_test_data.npy')
    y_test = np.load('/work/cse496dl/shared/hackathon/05/cifar10_test_labels.npy')
    train_data = imgs
    #original_train_data = imgs_large/255.
    test_data = test_data/255.
    #input_shape = train_data.shape[0]
    input_shape = (64,64,3)
    num_classes = 201
    print("image load complete")
    def model_arch():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        return model
    sess = tf.Session()
    keras.backend.set_session(sess)
    model_2 = model_arch()
    X_train = train_data
    Y_train = labels
    X_test = test_data
    Y_test = test_labels
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    x = tf.placeholder(tf.float32, shape=(None,32,32,3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    preds_2 = model_2(x)
    bim_params = {'eps_iter': 0.03,
              'nb_iter': 8,
              'clip_min': 0.,
              'clip_max': 1.}
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    fgsm2 = BasicIterativeMethod(model_2, sess=sess)
    adv_x_2 = fgsm2.generate(x, **bim_params)
    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x_2 = tf.stop_gradient(adv_x_2)
    preds_2_adv = model_2(adv_x_2)
    print("model and fgsm created successfully")
    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': 128}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    train_params = {
        'nb_epochs': 30,
        'batch_size': 20,
        'learning_rate': 0.01
    }
    print("train start")
    rng = np.random.RandomState([2017, 8, 30])
    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, X_train/255., Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng)
    model_2.save('bim_retrained_cifar.h5')
    print("train_end")
if __name__ == "__main__":
    tf.app.run()
