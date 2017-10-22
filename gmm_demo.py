# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:10:45 2017

@author: Heming Wang (20567431)
"""


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    normal_param = 1.0/(np.sqrt(2*np.pi*sigma**2))
    return normal_param*np.exp(- (x-mu)**2/(2*sigma**2) )


def responsibility(mix_sig, x, i, mus, sigmas):
    """Calculate the responsibility of the EM algorithm
    """
    
    pass


def preprocess(data_loc):
    """
    Converts original file data into list of floating point numbers
    """

    print("Preprocesing data...")
    
    # Get data into memory as list of lines
    with open(data_loc, 'r') as file:
        data_raw = file.read().splitlines()
        
    # Convert string float values to actual floats
    data = list(map(float, data_raw))
    
    return data


if __name__ == "__main__":
    # Preprocess
    # Source number
    k = 3
    data = preprocess("./em_data.txt")
    # print(data)


    # # Plot the signals
    # plt.plot(x, sig1)
    # plt.plot(x, sig2)
    # plt.plot(x, mix_sig)
    # # plt.show()
    
    x = np.asarray(data)
    N = len(x)
    # # original guess
    mus = [2, 15, 20]
    sigmas = [1.0, 1.0, 1.0]
    pi = [0.3, 0.4, 0.3]
    r = np.zeros([k, N])

        
    
    # # Use EM algorithm to maximize the prob
    
    # # Number of iterations
    M = 10

    for i in range(M):
        print("Step %d" % i)

        # Update the parameters
        
        # E Step
        mysum = 0
        for j in range(k):
            mysum += pi[j]*gaussian(x, mus[j], sigmas[j])
        for j in range(k):
            r[j] = pi[j]*gaussian(x, mus[j], sigmas[j]) / mysum

        # print("responsibility:")
        # print(r)

        # M step

        for j in range(k):
            pi[j] = 1.0/N*sum(r[j])
            mus[j] = 1.0/sum(r[j])*np.dot(r[j], x)
            sigmas[j] = 1.0/sum(r[j])*np.dot(r[j], (x - mus[j])**2)

        
        print("pi mu and signmas:")
        print(pi)
        print(mus)
        print(sigmas)
        print("---------\n")

        # print(pi)
        # print(i)



    # print(pi)
    # print(mus)
    # print(sigmas)





    data_max = max(x)
    data_min = min(x)
    # n, bins, patches = plt.hist(x, histtype = 'step')
    plt.subplot(2, 1, 1)
    plt.hist(x)

    myx = np.arange(data_min, data_max, 1.0/200)
    plt.subplot(2, 1, 2)
    
    restored_sig = 0
    for l in range(k):
        restored_sig += pi[l]*gaussian(myx, mus[l], sigmas[l])
    print(sum(restored_sig))
    plt.plot(myx, restored_sig)
    plt.show()
