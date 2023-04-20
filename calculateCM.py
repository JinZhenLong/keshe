import keras.backend as K
import tensorflow as tf
import numpy as np
#log10
def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
def calCM(sig):
    y_power = K.sqrt(K.mean(K.square(sig), axis=1))
    Vnorm=K.mean(sig,axis=1)/y_power
    cm=(y_power*Vnorm**3-1.52)/1.56
    return cm