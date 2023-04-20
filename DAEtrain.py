import sys
import glob

import keras.losses
import numpy as np
import scipy.io as scio
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
from DAE import CMnetEncoder, CMnetDecoder, CMnetAutoEncoder
from customLoss import CmLoss


N = 256

x = np.load('data/modChunks.npz')['data']
x = np.concatenate( [np.expand_dims(x.real, 1), np.expand_dims(x.imag, 1)], axis=1)
y = np.zeros(x.shape[0], dtype=np.float32)

xtrain = x[1000:]
ytrain = y[1000:]
xval = x[:1000]
yval = y[:1000]
print(xtrain.shape
,ytrain.shape
,xval.shape
,yval.shape)
encoder = CMnetEncoder(N)
decoder = CMnetDecoder(N)
autoencoder = CMnetAutoEncoder(N, encoder, decoder)
# loss1=keras.losses.huber(delta=1.0);
print(autoencoder.summary())
optimizer = Adam(learning_rate=0.001)
#0.0006
autoencoder.compile(loss=['huber_loss',CmLoss],loss_weights=[1,0.05] ,optimizer=optimizer)
#0.0006

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.01, cooldown=0, min_lr=1e-10))
callbacks.append(CSVLogger('trainingLog.csv', separator=',', append=False))
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=0, mode='auto'))

history = autoencoder.fit(xtrain, [xtrain, ytrain], validation_data=(xval, [xval,yval]), batch_size=256, epochs=20, callbacks=callbacks)
#
# a = history.history['loss']
# loss_data = 'loss_dataDAE.mat'
# scio.savemat(loss_data, {'loss_dataDAE':a})
#
#
# pyplot.plot(history.history['loss'])
# pyplot.show()
autoencoder.save('./train_model/for_plot/autoencoder5.hdf5')
encoder.save('./train_model/for_plot/encoder5.hdf5')
decoder.save('./train_model/for_plot/decoder5.hdf5')










