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
print("Finished Import")

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_integer('early_stop', 20, '')
flags.DEFINE_string('db', 'emodb', '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
flags.DEFINE_string('checkpoint_path', 'nips-2017-adversarial-learning-development-set/inception_v3.ckpt', 'Path to checkpoint for inception network.')
flags.DEFINE_string('input_dir', 'nips-2017-adversarial-learning-development-set/images/', 'Input directory with images.')
flags.DEFINE_string('output_dir', '', 'Output directory with images.')
flags.DEFINE_float('max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')
flags.DEFINE_integer('image_width', 28, 'Width of each input images.')
flags.DEFINE_integer('image_height', 28, 'Height of each input images.')
flags.DEFINE_float('eps', 2.0 * 16.0 / 255.0, '')
flags.DEFINE_integer('num_classes', 10, '')
flags.DEFINE_integer('num_ens', 10, '')
FLAGS = flags.FLAGS

def main(argv):

    print("Start Main")
    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    epochs = FLAGS.epochs
    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    learning_rate = FLAGS.lr
    early_stop = FLAGS.early_stop
    batch_size = FLAGS.batch_size
    reg_coeff = FLAGS.reg_coeff
    split = FLAGS.split
    master = FLAGS.master
    checkpoint_path = FLAGS.checkpoint_path
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    image_width = FLAGS.image_width
    image_height = FLAGS.image_height
    num_classes = FLAGS.num_classes
    eps = FLAGS.eps
    batch_shape = [batch_size, image_height, image_width, 3]
    num_ens = FLAGS.num_ens

    tf.logging.set_verbosity(tf.logging.INFO)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, image_width, image_height)
        x_test = x_test.reshape(x_test.shape[0], 1, image_width, image_height)
        input_shape = (1, image_width, image_height)
    else:
        x_train = x_train.reshape(x_train.shape[0], image_width, image_height, 1)
        x_test = x_test.reshape(x_test.shape[0], image_width, image_height, 1)
        input_shape = (image_width, image_height, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #Our model architecture for MNIST dataset
    def model_arch():
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        return model

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    sess = tf.Session()
    keras.backend.set_session(sess)
    
    x_noisy = util.add_gaussian_noise(x_train,0,64) #Add gaussian noise to all images
    preds_ens = np.zeros((x_test.shape[0],10)) #variable to store the predictions of each model in the ensemble (10)
    max_vote_ens = np.zeros(x_test.shape[0])  #variable to store Majority vote from all models in ensemble

    # for i in range(num_ens):
    #     model = model_arch() #Build a new model architecture for every model in the ensemble
    #     sub_imgs,sub_labels = util.subsample(x_noisy,y_train) #subsample from the entire data, bagging
    #     model.fit(sub_imgs, sub_labels, batch_size=batch_size,epochs=epochs,verbose=1) #train the model
    #     model.save("models/"+str(i)+".h5") #save the model
    #     ans = sess.run(tf.argmax(model.predict(x_test),axis=1))  #get the predictions of the model
    #     preds_ens[:,i]= ans.reshape((x_test.shape[0])) #store the predictions of this particular model(i) in ith column of pred_ens variable
    #     del model #erase the model

    # #Now the variable pred_ens consists of the predictions of all test_data for each model in ensemble.
    # #ith column contains predictions of ith model.
    # #go through every row
    # ens_acc = np.zeros(num_ens)
    # for i in range(num_ens):
    #     for j in range(preds_ens.shape[0]):
    #         b= Counter(preds_ens[j][0:i+1]) #get the entire row which consists of predictions for that particular instance from all models.
    #         max_vote_ens[j] = b.most_common(1)[0][0] #get the maximum vote i.e which number has more frequency.
    #     ens_acc_i = sess.run(tf.reduce_mean(tf.cast(tf.equal(max_vote_ens, tf.argmax(y_test, axis=1)) , tf.float32)))
    #     ens_acc[i] = ens_acc_i #accuracy of ensemble
    #     #TODO print the nonperturbed test accuracy to the output file.

    # #Build a model for normal training on the entire noisy data.
    # model = model.model_arch()
    # model.fit(x_noisy, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    # acc = model.evaluate(x_test, y_test, verbose=0)
    # acc_noisy_normal = acc[1] #accuracy of normal model on noisy train data
    # del model

    #Build a new model for normal training (without ensemble) on entire train data (with out bagging and noise).
    model = model_arch()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    acc = model.evaluate(x_test, y_test, verbose=0)
    model.save("models/original_model.h5")

    #accuracy of normal model
    acc_normal = acc[1]

    #generate fgsm adversarial examples on test_data
    adv_fgsm = util.fgsm_attack(x_test,model,sess)
    acc_fgsm = model.evaluate(adv_fgsm, y_test, verbose=0)
    acc_fgsm = acc_fgsm[1]  #accuracy of normal model on fgsm adversarial examples

    #generate bim adversarial examples on test_data
    adv_bim = util.bim_attack(x_test,model,sess)
    acc_bim = model.evaluate(adv_bim,y_test,verbose=0)
    acc_bim = acc_bim[1] #accuracy of normal model on bim adversarial examples

    #generate lbfgs adversarial examples on test_data
    # The target is chosen as 6
    adv_lbfgs = util.lbfgs_attack(x_test,model,sess,6)
    acc_lbfgs = model.evaluate(adv_lbfgs,y_test,verbose=0)
    acc_lbfgs = acc_lbfgs[1] #accuracy of normal model on lbfgs adversarial examples

    preds_ens_fgsm = np.zeros((x_test.shape[0],10)) #variable to store the predictions of each model in the ensemble (10) for fgsm adversarial examples
    max_vote_ens_fgsm = np.zeros(x_test.shape[0]) #variable to store Majority vote from all models in ensemble for fgsm adversarial examples
    preds_ens_bim = np.zeros((x_test.shape[0],10)) #variable to store the predictions of each model in the ensemble (10) for bim adversarial examples
    max_vote_ens_bim = np.zeros(x_test.shape[0]) #variable to store Majority vote from all models in ensemble for bim adversarial examples
    preds_ens_lbfgs = np.zeros((x_test.shape[0],10)) #variable to store the predictions of each model in the ensemble (10) for lbfgs adversarial examples
    max_vote_ens_lbfgs = np.zeros(x_test.shape[0]) #variable to store Majority vote from all models in ensemble for lbfgs adversarial examples

    del model

    for i in range(num_ens):
        model = load_model("models/"+str(i)+".h5")
        #get predictions of model i for fgsm adversarial examples
        ans = sess.run(tf.argmax(model.predict(adv_fgsm),axis=1))
        preds_ens_fgsm[:,i]= ans.reshape((adv_fgsm.shape[0]))
         #get predictions of model i for bim adversarial examples
        ans = sess.run(tf.argmax(model.predict(adv_bim),axis=1)) 
        preds_ens_bim[:,i]= ans.reshape((adv_bim.shape[0]))
        #get predictions of model i for lbfgs adversarial examples
        ans = sess.run(tf.argmax(model.predict(adv_lbfgs),axis=1)) 
        preds_ens_lbfgs[:,i]= ans.reshape((adv_lbfgs.shape[0]))
        del model

    #Now the variable pred_ens consists of the predictions of all fgsm adversarial test_data for each model in ensemble.
    #ith column contains predictions of ith model.
    #go through every row
    ens_acc_fgsm = np.zeros(num_ens)
    for i in range(num_ens):
        for j in range(preds_ens_fgsm.shape[0]):
            b= Counter(preds_ens_fgsm[j][0:i+1])  #get the entire row which consists of predictions for that particular instance from all models.
            max_vote_ens_fgsm[j] = b.most_common(1)[0][0] #get the maximum vote i.e which number has more frequency.
        #accuracy of ensemble
        ens_acc_fgsm_i = sess.run(tf.reduce_mean(tf.cast(tf.equal(max_vote_ens_fgsm, tf.argmax(y_test, axis=1)) , tf.float32)))
        ens_acc_fgsm[i] = ens_acc_fgsm_i

    #Now the variable pred_ens consists of the predictions of all bim adversarial test_data for each model in ensemble.
    #ith column contains predictions of ith model.
    #go through every row
    ens_acc_bim = np.zeros(num_ens)
    for i in range(num_ens):
        for j in range(preds_ens_bim.shape[0]):
            b= Counter(preds_ens_bim[j][0:i+1])
            max_vote_ens_bim[j] = b.most_common(1)[0][0]
        #accuracy of ensemble on bim_adv
        ens_acc_bim_i = sess.run(tf.reduce_mean(tf.cast(tf.equal(max_vote_ens_bim, tf.argmax(y_test, axis=1)) , tf.float32)))
        ens_acc_bim[i] = ens_acc_bim_i

    #Now the variable pred_ens consists of the predictions of all lbfgs adversarial test_data for each model in ensemble.
    #ith column contains predictions of ith model.
    #go through every row
    ens_acc_lbfgs = np.zeros(num_ens)
    for i in range(num_ens):
        for i in range(preds_ens_lbfgs.shape[0]):
            b= Counter(preds_ens_lbfgs[j][0:i+1])
            max_vote_ens_lbfgs[j] = b.most_common(1)[0][0]
        #accuracy of ensemble on lbfgs_adv
        ens_acc_lbfgs_i = sess.run(tf.reduce_mean(tf.cast(tf.equal(max_vote_ens_lbfgs, tf.argmax(y_test, axis=1)) , tf.float32)))
        ens_acc_lbfgs[i] = ens_acc_lbfgs_i


    #-----------------------------------Adversarial Training--------------------------------------------------------------
    #first adversarial examples are generated using train_data, then the model is trained on train_data+adv_train_data.
    #Then the model is tested on normal test_data, then the model is tested on adversarial_test_data.
    #So, we are generating the adversarial examples twice both on train and test data.

    model = load_model("models/original_model.h5")
    wrap = KerasModelWrapper(model)

    #generate adversarial examples on train data.
    adv_fgsm_train = util.fgsm_attack(x_train,model,sess)
    adv_bim_train = util.bim_attack(x_train,model,sess)
    adv_lbfgs_train = util.lbfgs_attack(x_train,model,sess,6)
    train_plus_adv_fgsm = np.concatenate([x_train,adv_fgsm_train])
    y_train_plus_adv_fgsm = np.concatenate([y_train,y_train])
    train_plus_adv_bim = np.concatenate([x_train,adv_bim_train])
    y_train_plus_adv_bim = np.concatenate([y_train,y_train])
    train_plus_adv_lbfgs = np.concatenate([x_train,adv_lbfgs_train])
    y_train_plus_adv_lbfgs = np.concatenate([y_train,y_train])
    del model

    #FGSM TRAINING
    #build a fresh model for fgsm training
    model = model_arch()
    wrap = KerasModelWrapper(model)
    model.fit(train_plus_adv_fgsm, y_train_plus_adv_fgsm, batch_size=batch_size, epochs=epochs, verbose=1)
    fgsm_acc_train = model.evaluate(x_test,y_test,verbose=0)
    fgsm_acc_train[1] #Accuracy of adversarially trained model on clean examples

    #generate adversarial examples for adversarially trained model on test_data
    adv_fgsm_test = util.fgsm_attack(x_test,model,sess)
    fgsm_adv_acc_train = model.evaluate(adv_fgsm_test,y_test,verbose=0)
    fgsm_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images

    del model
    
    #BIM TRAINING
    #build a fresh model for bim training
    model = model_arch()
    wrap = KerasModelWrapper(model)
    model.fit(train_plus_adv_bim, y_train_plus_adv_bim, batch_size=batch_size, epochs=epochs, verbose=1)
    bim_acc_train = model.evaluate(x_test,y_test,verbose=0)
    bim_acc_train[1] #Accuracy of adversarially trained model on clean examples

    #generate adversarial examples for adversarially trained model on test_data
    adv_bim_test = util.bim_attack(x_test,model,sess)
    bim_adv_acc_train = model.evaluate(adv_bim_test,y_test,verbose=0)
    bim_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images

    del model

    #LBFGS TRAINING
    #build a fresh model for lbfgs training
    model = model_arch()
    wrap = KerasModelWrapper(model)
    model.fit(train_plus_adv_lbfgs, y_train_plus_adv_lbfgs, batch_size=batch_size, epochs=epochs, verbose=1)
    lbfgs_acc_train = model.evaluate(x_test,y_test,verbose=0)

    #Accuracy of adversarially trained model on clean examples
    lbfgs_acc_train[1]
    adv_lbfgs_test = util.lbfgs_attack(x_test,model,sess,6)
    lbfgs_adv_acc_train = model.evaluate(adv_lbfgs_test,y_test,verbose=0)
    lbfgs_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images
    
    del model

if __name__ == "__main__":
    tf.app.run()