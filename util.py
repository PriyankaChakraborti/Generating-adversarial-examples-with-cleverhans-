import os
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
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import mnist

def split_data(data, labels, proportion):
    """                                                                                                                                                                                                                                                                                                 
    """
    size = data.shape[0]
    np.random.seed(42)
    indicies = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[indicies[:split_idx]], data[indicies[split_idx:]], labels[indicies[:split_idx]], labels[indicies[split_idx:]]

def transform_labels(labels):
    new_labels = []
    for l in labels:
        t_label = np.zeros(10)
        t_label[(int(l)-1)] = 1.
        new_labels.append(t_label)
    return np.array(new_labels)

def onehot(data):
    data_final = np.zeros((data.shape[0],10))
    data_final[np.arange(data.shape[0]),data.astype(int)]=1
    return data_final

def psnr(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    psnr = 20*np.log10(255) - 10*np.log10(err)
    return psnr

def subsample(training_images,training_labels, ratio=0.8):
    shuffle_index = np.random.permutation(len(training_labels))
    training_images = training_images[shuffle_index]
    training_labels = training_labels[shuffle_index]
    sample = list()
    sample_labels = list()
    n_sample = round(training_images.shape[0] * ratio)
    while len(sample) < n_sample:
        index = randrange(training_images.shape[0])
        sample.append(training_images[index,:,:,:])
        sample_labels.append(training_labels[index])
    return np.asarray(sample),np.asarray(sample_labels)

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    # Limit to first 20 images for this example
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png')))[:20]:
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')

def add_gaussian_noise(X_train, mean, stddev):
    ''' 
    INPUT:  (1) 4D numpy array: all raw training image data, of shape 
                (#imgs, #chan, #rows, #cols)
            (2) float: the mean of the Gaussian to sample noise from
            (3) float: the standard deviation of the Gaussian to sample
                noise from. Note that the range of pixel values is
                0-255; choose the standard deviation appropriately. 
    OUTPUT: (1) 4D numpy array: noisy training data, of shape
                (#imgs, #chan, #rows, #cols)
    '''
    n_imgs = X_train.shape[0]
    n_chan = X_train.shape[3]
    n_rows = X_train.shape[1]
    n_cols = X_train.shape[2]
    if stddev == 0:
        noise = np.zeros((n_imgs, n_rows, n_cols,n_chan))
    else:
        noise = np.random.normal(mean, stddev/255., 
                                 (n_imgs,n_rows, n_cols,n_chan))
    noisy_X = X_train + noise
    clipped_noisy_X = np.clip(noisy_X, 0., 1.)
    return clipped_noisy_X

def fgsm_attack(train_data,model,sess):
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate_np(train_data, **fgsm_params)
    return adv_x

def bim_attack(train_data,model,sess):
    wrap = KerasModelWrapper(model)
    bim = BasicIterativeMethod(wrap, sess=sess)
    bim_params = {'eps_iter': 0.01,
              'nb_iter': 10,
              'clip_min': 0.,
              'clip_max': 1.}
    adv_x = bim.generate_np(train_data, **bim_params)
    return adv_x

def lbfgs_attack(train_data,model,sess,tar_class):
    wrap = KerasModelWrapper(model)
    lbfgs = LBFGS(wrap,sess=sess)
    one_hot_target = np.zeros((train_data.shape[0], 10), dtype=np.float32)
    one_hot_target[:, tar_class] = 1
    adv_x = lbfgs.generate_np(train_data, max_iterations=10,
                                        binary_search_steps=3,
                                        initial_const=1,
                                        clip_min=-5, clip_max=5,
                                        batch_size=1, y_target=one_hot_target)
    return adv_x

def noisy(noise_typ,image):
    #Gaussian
   if noise_typ == 1:
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy.astype('uint8')
    #Salt and Pepper
   elif noise_typ == 2:
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(i - 1,0, int(num_salt))
              for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint( i - 1,0, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    #Poisson
   elif noise_typ == 3:
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    #speckle
   elif noise_typ ==4:
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy.astype('uint8')

def subsample(training_images,training_labels, ratio=0.8):
    shuffle_index = np.random.permutation(len(training_labels))
    training_images = training_images[shuffle_index]
    training_labels = training_labels[shuffle_index]
    sample = list()
    sample_labels = list()
    n_sample = round(training_images.shape[0] * ratio)
    while len(sample) < n_sample:
        index = randrange(training_images.shape[0])
        sample.append(training_images[index,:,:,:])
        sample_labels.append(training_labels[index])
    return np.asarray(sample),np.asarray(sample_labels)

def load_training_images(training_image_dir):

    image_index = 0
    
    images = np.ndarray(shape=(500*200, 64,64,3))
    names = []
    labels = []                       
    
    # Loop through all the types directories
    for type in os.listdir(training_image_dir):
        if os.path.isdir(training_image_dir + type + '/images/'):
            type_images = os.listdir(training_image_dir + type + '/images/')
            # Loop through all the images of a type directory
            #batch_index = 0;
            #print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(training_image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = np.asarray(Image.open(image_file),)
                #print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (64, 64, 3)):
                    images[image_index, :,:,:] = image_data
                    
                    labels.append(type)
                    names.append(image)
                    
                    image_index += 1
                    #batch_index += 1
                #if (batch_index >= batch_size):
                 #   break;
    labels = np.asarray(labels)
    names = np.asarray(names)
    return (images[0:len(labels)].astype('uint8'), labels, names)

def rescale(img):
    print(img)
    img = Image.fromarray(img)
    basewidth =299
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    return img

def load_val_images(val_image_dir):
    image_index = 0
    images = np.ndarray(shape=(500*20, 64,64,3))
    names = []
    labels = [] 
    df = pd.read_csv(val_image_dir+'val_annotations.txt',header=None,sep='\t')
    for image in os.listdir(val_image_dir+'/images/'):
        image_file = os.path.join(val_image_dir+ '/images/', image)

                # reading the images as they are; no normalization, no color editing
        image_data = np.asarray(Image.open(image_file),)
                #print ('Loaded Image', image_file, image_data.shape)
        if (image_data.shape == (64, 64, 3)):
            images[image_index, :,:,:] = image_data
            names.append(image)
            image_index += 1
            labels.append(df[df[0] == image][1].values[0])
    names = np.asarray(names)
    labels =np.asarray(labels)
    return (images[0:len(labels)].astype('uint8'), labels)

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x) 
  predictions = Dense(nb_classes, activation='softmax')(x) 
  model = Model(input=base_model.input, output=predictions)
  return model

def setup_to_finetune(model):
   """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top 
      layers.
   note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in 
         the inceptionv3 architecture
   Args:
     model: keras model
   """
   NB_IV3_LAYERS_TO_FREEZE = 172
   for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
      layer.trainable = False
   for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
      layer.trainable = True
   model.compile(optimizer=SGD(lr=0.001, momentum=0.9),   
                 loss='categorical_crossentropy')

def image_loader(batch_size,imgs,labels):
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < imgs.shape[0]:
            limit = min(batch_end,imgs.shape[0])
            x = imgs[batch_start:limit]
            imgs_large = np.ndarray(shape= [x.shape[0],299,299,3])
            for i in range(x.shape[0]):
                imgs_large[i,:,:,:] = rescale(x[i])
            imgs_large=imgs_large.astype('uint8')
            imgs_noisy = np.ndarray(shape= imgs_large.shape)
            for i in range(imgs_large.shape[0]):
                imgs_noisy[i,:,:,:] = noisy(1,imgs_large[i])
            imgs_noisy=imgs_noisy.astype('uint8')
            x = imgs_noisy
            y = labels[batch_start:limit]
            #for i in y:
            #    ind = all_labels.index(i)
            #    ind_sub = list(y).index(i)
            #    y[ind_sub] = ind
            y = keras.utils.to_categorical(y, num_classes=201)
            yield(x/255.,y)
            batch_start+=batch_size
            batch_end+=batch_size

def image_loader_original(batch_size,imgs,labels):
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < imgs.shape[0]:
            limit = min(batch_end,imgs.shape[0])
            x = imgs[batch_start:limit]
            imgs_large = np.ndarray(shape= [x.shape[0],299,299,3])
            for i in range(x.shape[0]):
                imgs_large[i,:,:,:] = rescale(x[i])
            imgs_large=imgs_large.astype('uint8')
            y = labels[batch_start:limit]
            y = keras.utils.to_categorical(y, num_classes=201)
            yield(x/255.,y)
            batch_start+=batch_size
            batch_end+=batch_size