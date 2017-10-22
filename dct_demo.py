import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct
from scipy.io import wavfile
from scipy import signal


fs, data = wavfile.read('./sound/speech-female.wav')
# number of points
# N = len(data)
# # Time
# T = float(N)/fs

# # number of frames
# N_frame = 30
# frame_length = T/N_frame


# plt.plot(dct(data[10*fs*frame_length:11*fs*frame_length], 1))
# plt.show()

f, p = signal.periodogram(data, fs)
plt.semilogy(f, np.sqrt(p))
plt.show()