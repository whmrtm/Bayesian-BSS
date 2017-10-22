# Gaussian mixture model of the audio spectrum

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct, fftshift
from scipy.stats import rv_discrete
from scipy.io import wavfile

def gaussian(x, mu, sigma_sq):
    normal_param = 1.0/(np.sqrt(2*np.pi*sigma_sq))
    return normal_param*np.exp(- (x-mu)**2/(2*sigma_sq) )





if __name__ == "__main__":

    # N = 64
    # dw = 0.3
    # w = np.linspace(0,N*dw-dw,N)
    # sig = 200 + (np.arange(N)-N/2)**2*np.random.random(N)


    fs, data = wavfile.read('./sound/speech-female.wav')

    S = fftshift(fft(data[:500]))
    # normalized spectrum
    F = abs(S)/sum(abs(S))
    N = len(S)

    F_bin = np.linspace(0,N,N)
    plt.bar(F_bin, F, align='center')
    # plt.xticks([i*8*dw for i in range(N/8)]+[N*dw-dw/2])
    # plt.xlim(-dw/2,N*dw-dw/2)
    # plt.show()

    # For a single time frame t, do the transform and obtain the spectrum F(fi)

    # Initilization
    # Number of components in the Gaussian mixtures
    M = 10


    # Monto Carlo method

    # Number of points
    # N = 1000
    # Generate f bins data according to the distribution of F
    # f_distribute = rv_discrete(name = 'f_distribute', \
    # values = (F_bin, F))
    # rand_size = 1000
    # f_data = f_distribute.rvs(size = rand_size)
    # f_data = F_bin[f_data]


    # Mu, sigma_sq and omega
    mu = np.ones(M)
    for m in range(M):
        mu[m] = max(F_bin)*(m+0.5)/M + 0.7

    sigma_sq = np.ones(M)
    sigma_sq *= (float(N)**2)/(float(M)**2)

    omega = np.ones(M)
    omega /= float(M)

    r = np.zeros([M, N])

    num_iter = 30
    
    print(omega)
    print(mu)
    print(sigma_sq)


    for i in range(num_iter):

        # E step, calculate responsibilities
        # P(omega_j | g_i, theta)
        # Given a spectrum f
      
        def p_log(F_bin, mu, sigma_sq):

            p_log = np.log(1.0/np.sqrt(2*np.pi*sigma_sq)) - \
            ((F_bin - mu)**2 + 1.0/12.0)/(2*sigma_sq)
            return p_log

        r_sum = 0
        for j in range(M):
            r_sum += omega[j] * np.exp(p_log(F_bin, mu[j], sigma_sq[j]))
  
        for j in range(M):
            r[j] = (omega[j] * np.exp(p_log(F_bin, mu[j], sigma_sq[j]))) / r_sum
        

        # M step
        # print(r[0])
        # print(r[1])

        for j in range(M):
            mu[j] = np.dot(F, r[j]*F_bin) / (np.dot(F, r[j]))
            sigma_sq[j] = np.dot(F*r[j], ((F_bin-mu[j])**2 + 1.0/12.0)) / np.dot(F, r[j]) 
            omega[j] = np.dot(F, r[j])
        omega_sum = sum(omega)
        omega /= omega_sum


        print("\n---------Step %d ---------" % i)

        print(omega)
        print(mu)
        print(sigma_sq)
    restored_spec = np.zeros(N)
    for j in range(M):
        restored_spec += omega[j]*gaussian(F_bin, mu[j], sigma_sq[j])
    plt.plot(F_bin, restored_spec)
    print(sum(F))
    print(sum(restored_spec))
    plt.show()