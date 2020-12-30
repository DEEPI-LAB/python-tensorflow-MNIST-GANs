
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:06:04 2020

@author: pod LAB. Kim Jongwon
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy import io

mnist_x = io.loadmat('train_input.mat')['images']
minst_y = io.loadmat('train_output.mat')['y']
mnist_x = mnist_x.astype('float32')

# Generator
Generator = tf.keras.Sequential([
    tf.keras.layers.Input(256,30),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid')])

# Discriminator
Discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(784),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

Doptimizer = tf.keras.optimizers.Adam(0.001)
Goptimizer = tf.keras.optimizers.Adam(0.001)

def get_noise(batch_size,n_noise):
    return tf.random.normal([batch_size,n_noise])

@tf.function
def train_step(inputs):

    with tf.GradientTape() as t1, tf.GradientTape() as t2:
        
        G = Generator(get_noise(30,256))
    
        Z = Discriminator(G)
        R = Discriminator(inputs)   
        loss_D = -tf.reduce_mean(tf.math.log(R) + tf.math.log(1 - Z))
        loss_G = -tf.reduce_mean(tf.math.log(Z))
          
    Dgradients = t1.gradient(loss_D, Discriminator.trainable_variables)
    Doptimizer.apply_gradients(zip(Dgradients, Discriminator.trainable_variables))
    
    Ggradients = t2.gradient(loss_G,Generator.trainable_variables)
    Goptimizer.apply_gradients(zip(Ggradients, Generator.trainable_variables))    
        
   
total_batch = int(60000/30) 
        
for epoch in tf.range(15):
    k = 0
    for i in tf.range(total_batch):
        batch_input = mnist_x.T[i*30:(i+1)*30]
    
        inputs = tf.Variable([batch_input],tf.float32)
        train_step(inputs)
        print(k)
        k = k + 1

        if k%100 == 0:
            G = Generator(get_noise(10,256))
        
            fig, ax = plt.subplots(1,10 ,figsize=(10, 1))
                
            for j in range(10):
                ax[j].set_axis_off()
                ax[j].imshow(np.reshape(G[j], (28, 28)).T,cmap='gray')
            plt.pause(0.001)
            plt.show()
            


