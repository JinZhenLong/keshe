import tensorflow as tf
import keras.backend as K
from keras.backend import mean, var, max, abs, square, sqrt
from calculateCM import calCM
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
def CmLoss(y_ture,y_pred):
    return calCM(y_pred)
