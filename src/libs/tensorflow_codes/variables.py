import tensorflow as tf
import numpy as np

v1 = tf.Variable(tf.constant(2.0,shape=[4]),dtype='float32')
v2 = tf.Variable(np.ones(shape=[4,3]),dtype='float32')
v3 = tf.Variable(tf.keras.initializers.RandomNormal()(shape=[3,4,5]),dtype='float32')

print("v1",v1)
print("v2",v2)
print("v3",v3)
