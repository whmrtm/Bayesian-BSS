# Gaussian mixture model of the audio spectrum

import numpy as np
# from numpy.fft import rfft, fftfreq
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct, fftshift
from scipy.stats import rv_discrete
from scipy.io import wavfile

class gmm_model():
    """ Given a normalized spectrum for a single time frame, 
    the class return the the GMM mixture model paramters
    """
    def __init__(self, K, F):
        # number of Gaussians
        self.K = K
        # number of frequency bins in the time frame
        self.N = len(F)
        # normalized frequency spectrum
        self.F = F
        # Frequency bins
        self.F_bin = np.linspace(-np.floor(self.N//2), \
        np.floor(self.N//2), self.N)

        # GMM paramters, here we set all the mus to zero
        self.sigma_sq = np.ones(K)
        for k in range(K):
            self.sigma_sq[k] *= float(k+1)/float(K) * (float(self.N)**2)/(float(K)**2)
        self.omega = np.ones(K)
        self.omega /= float(K)

    def Gaussian(self, x, mu, sigma_sq):
        """ Gaussian function
        """
        normal_param = 1.0/(np.sqrt(2*np.pi*sigma_sq))
        return normal_param*np.exp(- (x-mu)**2/(2*sigma_sq) )


    def update_gmm(self, sigma_sq, omega):
        """ Update gmm parameters
        """
        self.sigma_sq = sigma_sq
        self.omega = omega    

    def em(self, iter_steps = 30):
        """ Apply the em algorithm to estimate GMM paramteres
        """

        K = self.K
        N = self.N
        F = self.F
        F_bin = self.F_bin
        sigma_sq = self.sigma_sq
        omega = self.omega

        # responsibilities
        r = np.zeros([K, N])
        def p_log(F_bin, mu, sigma_sq):
            """ Helper function for calculating responsibilities
            """
            p_log = np.log(1.0/np.sqrt(2*np.pi*sigma_sq)) - \
            ((F_bin - mu)**2 + 1.0/12.0)/(2*sigma_sq)
            return p_log

        for i in range(iter_steps):
            # print("\n----- Step %d -----" % i)
            # E step
            r_sum = 0
            for j in range(K):
                r_sum += omega[j] * np.exp(p_log(F_bin, 0, sigma_sq[j]))
            for j in range(K):
                r[j] = (omega[j] * np.exp(p_log(F_bin, 0, sigma_sq[j]))) / r_sum
        
            # M step
            for j in range(K):
                sigma_sq[j] = np.dot(F*r[j], ((F_bin)**2 + 1.0/12.0)) / np.dot(F, r[j]) 
                omega[j] = np.dot(F, r[j])
            omega_sum = sum(omega)
            omega /= omega_sum
            
            # print("sigma_sq, omega")
            # print(sigma_sq, omega)            
        self.update_gmm(sigma_sq, omega)
    
    def bar_plot(self):
        """ Plot the bar graph of the frequency distribution
        """
        plt.bar(self.F_bin, self.F, align='center')

    def gmm_plot(self):
        """ Plot the result
        """
        K = self.K
        N = self.N
        F = self.F
        F_bin = self.F_bin
        sigma_sq = self.sigma_sq
        omega = self.omega

        restored_spec = np.zeros(N)
        self.bar_plot()
        for j in range(K):
            restored_spec += omega[j]*self.Gaussian(F_bin, 0, sigma_sq[j])
        plt.plot(F_bin, restored_spec)
        print(sum(self.F))
        print(sum(restored_spec))
        plt.show()

    def output_gmm(self):
        """ Run the iteration and return results
        """
        self.em()
        # print(self.sigma_sq)
        # print(self.omega)
        return self.sigma_sq, self.omega


if __name__ == "__main__":
    fs, data = wavfile.read('./sound/speech-female.wav')
    S = fftshift(fft(data[:500]))
    # normalized spectrum
    F = abs(S)/sum(abs(S))
    F_gmm = gmm_model(12, F)
    # plt.plot(F)
    # plt.show()
    sigma_sq, omega = F_gmm.output_gmm()
    print(sigma_sq, omega)
    F_gmm.gmm_plot()



