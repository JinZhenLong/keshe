# import sys
import numpy as np
# sys.path.append("..")
from DAE import CMnetEncoder, CMnetDecoder
import cmath
import scipy.io as scio
from time import time
N = 256


x = np.load('data/modChunks.npz')['data']
xval = x[0:1000]


# data1 = 'data_after_4QAM.mat'
# scio.savemat(data1, {'data_after_4QAM':xval})



xsig = np.fft.ifft(xval, n=256, axis=1)*cmath.sqrt(N)


# data2 = 'data_original_after_ifft.mat'
# scio.savemat(data2, {'data_original_after_ifft':xsig})



xvalrect = np.concatenate( [np.expand_dims(xval.real, 1), np.expand_dims(xval.imag, 1)], axis=1)


# Encoding

encoder = CMnetEncoder(N)
encoder.load_weights('train_model/for_plot/encoder10.hdf5')



xenc = encoder.predict(xvalrect, batch_size=256)

xhidd = xenc[:, 0, :] + 1j * xenc[:, 1, :]


data3 = 'data_paprnet_before_ifft10.mat'
scio.savemat(data3, {'data_paprnet_before_ifft10':xhidd})

signal = xsig

# dataFile = 'encode_after_fft.mat'
# data = scio.loadmat(dataFile)
# x = data['encode_after_fft']

xenc = np.concatenate( [np.expand_dims(x.real, 1), np.expand_dims(x.imag, 1)], axis=1)



# Decoding
decoder = CMnetDecoder(N)
decoder.load_weights('train_model/for_plot/decoder10.hdf5')


xdec = decoder.predict(xenc, batch_size=256)

xest = xdec[:, 0, :] + 1j * xdec[:, 1, :]




data4 = 'data_paprnet_after_fft10.mat'
scio.savemat(data4, {'data_paprnet_after_fft10':xest})


# data5 = 'data_original_after_fft.mat'
# scio.savemat(data5, {'data_original_after_fft':signal})



#
