# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:18:47 2020
@author: pod LAB. Kim Jongwon
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pytictoc import TicToc
from scipy import io


# fashion_mnist = tf.keras.datasets.fashion_mnist
# (train_images,_),(_, _) = fashion_mnist.load_data()
# mnist_x  = (train_images.astype('float32') - 127.5) /127.5
mnist_x = io.loadmat('train_input.mat')['images']
minst_y = io.loadmat('train_output.mat')['y']
mnist_x = mnist_x.astype('float32')
mnist_x = np.reshape((mnist_x*2-1) ,(28,28,60000)).T

batch = 10
# Generator Net
Generator = tf.keras.Sequential([
    tf.keras.layers.Input(100,None),
    tf.keras.layers.Dense(7*7*512,use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((7,7,512)),
    tf.keras.layers.Conv2DTranspose(512, (4,4), padding='same',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same',use_bias=False),
    ])

Generator.summary()

# Discriminator Net
Discriminator = tf.keras.Sequential([
    tf.keras.layers.Input((28,28,1),None),
    tf.keras.layers.Conv2D(32, (4,4),strides=(2,2) ,padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64,(4,4),strides=(2,2) ,padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128,(4,4),strides=(2,2) ,padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(256,(4,4),strides=(2,2) ,padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(512,(4,4),strides=(2,2) ,padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation='sigmoid')])

Discriminator.summary()

Doptimizer = tf.keras.optimizers.Adam(0.00001)
Goptimizer = tf.keras.optimizers.Adam(0.00001)

def get_noise(batch_size,n_noise):
    return tf.random.normal([batch_size,n_noise])

@tf.function
def train_step(inputs):

    with tf.GradientTape() as t1, tf.GradientTape() as t2:
        
        G = Generator(get_noise(batch,100))
    
        Z = Discriminator(G, training=True)
        R = Discriminator(inputs, training=True)   
        loss_D = -tf.reduce_mean(tf.math.log(R) + tf.math.log(1 - Z))
        loss_G = -tf.reduce_mean(tf.math.log(Z))
          
    Dgradients = t1.gradient(loss_D, Discriminator.trainable_variables)
    Doptimizer.apply_gradients(zip(Dgradients, Discriminator.trainable_variables))
    
    Ggradients = t2.gradient(loss_G,Generator.trainable_variables)
    Goptimizer.apply_gradients(zip(Ggradients, Generator.trainable_variables))    
        
   
total_batch = int(60000/batch) 

t = TicToc()   
t.tic()  
for epoch in tf.range(1):
    k = 0
    for i in tf.range(total_batch):
        batch_input = mnist_x[i*batch:(i+1)*batch].T
    
        inputs = tf.transpose(tf.Variable([batch_input],tf.float32))
        train_step(inputs)
        print(k)
        k = k + 1

        # if k%100 == 0:
        #     G = Generator(get_noise(10,100))
        
        #     fig, ax = plt.subplots(1,10 ,figsize=(10, 1))
                
        #     for j in range(10):
        #         ax[j].set_axis_off()
        #         ax[j].imshow(G[j],cmap='gray')
        #     plt.pause(0.001)
        #     plt.show()
t.toc()