import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, ReLU, Lambda, Concatenate, Flatten, Add
import math
#gennerate Gaussian noise
def GaussianNoise(x, sigma):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)
    return x + noise

def IFFT(sig, name=None):
    N = int(sig.shape[-1])
    return Lambda(tf.signal.ifft, name=name, output_shape=(1,N))(sig)
def FloatToComplex(sig, name=None):
    '''
    给两个实数，转化成复数
    :param sig:
    :param name:
    :return:
    '''
    N = int(sig.shape[-1])
    return Lambda(lambda x : tf.complex(x[:, 0:1, :], x[:, 1:2, :]), name=name, output_shape=(1,N))(sig)

def ComplexToFloat(sig, name=None):
    N = int(sig.shape[-1])
    return Concatenate(name=name, axis=-2)([Lambda(tf.math.real, output_shape=(1, N), name=name+'_Re')(sig), Lambda(tf.math.imag, output_shape=(1, N), name=name+'_Im')(sig)])

def CMnetEncoder(N):

    enc_in = Input((2, N), name="encoder_input")
    enc_in_noised=GaussianNoise(enc_in,0.1)
    h1 = Dense(N*2, activation='relu', name='Dense1')(enc_in_noised)
    h1 = BatchNormalization(name='DenseBN1')(h1)

    h2 = Dense(N*2, activation='relu', name='Dense2')(h1)
    h2 = BatchNormalization(name='DenseBN2')(h2)

    h3 = Dense(N*2, activation='relu', name='Dense3')(h2)
    h3 = BatchNormalization(name='DenseBN3')(h3)

    h4 = Dense(N*2, activation='relu', name='Dense4')(h3)
    h4 = BatchNormalization(name='DenseBN4')(h4)


    h6 = Dense(N, activation='relu', name='Dense6')(h4)
    h6 = BatchNormalization(name='DenseBN6')(h6)

    enc_out = Lambda(lambda x: x**3, name='encoder_output')(h6)

    return Model(inputs=[enc_in], outputs=[enc_out], name="DAECMnet_Encoder")    # 包含从 enc_in 到 enc_out 的计算的所有网络层

def CMnetDecoder(N):

    dec_in = Input((2,N), name="decoder_input")

    h7 = Dense(N*2, activation='relu', name='Dense7')(dec_in)
    h7 = BatchNormalization(name='DenseBN7')(h7)

    h8 = Dense(N*2, activation='relu', name='Dense8')(h7)
    h8 = BatchNormalization(name='DenseBN8')(h8)

    h9 = Dense(N*2, activation='relu', name='Dense9')(h8)
    h9 = BatchNormalization(name='DenseBN9')(h9)

    h10 = Dense(N*2, activation='tanh', name='Dense10')(h9)
    h10 = BatchNormalization(name='DenseBN10')(h10)


    dec_out = Dense(N, activation='linear', name='decoder_output')(h10)

    return Model(inputs=[dec_in], outputs=[dec_out], name="DAECMnet_Decoder")

def CMnetAutoEncoder(N, enc, dec):
    

    enc_in = enc.input
    enc_out = enc(enc_in)
    dec_out = dec(enc_out)

    # taking ifft of encoder output - used to minimize (CM)
    cmplx = FloatToComplex(enc_out, name='EncoderOut-FloatToComplex')
    ifft = IFFT(cmplx, name='%d-IFFT' % N)
    ifft = ComplexToFloat(ifft, name='%d-IFFT-ComplexToFloat' % N)
    print(enc_in.shape)
    print(enc_out.shape)
    print(dec_out.shape)

    return Model(inputs=[enc_in], outputs=[dec_out, ifft])
