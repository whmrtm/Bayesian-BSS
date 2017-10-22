from gmm_model import *
from stft import *



if __name__ == "__main__":
    file_name = "./sound/male-female.wav"
    fs, x = wavfile.read(file_name)
    X = stft(x)

    frames_num = X.shape[0]

    s1_spectrum = np.zeros(X.shape, dtype=complex)
    s2_spectrum = np.zeros(X.shape, dtype=complex)

    for frame in range(frames_num):
        F = X[frame]
        F = np.concatenate((F[::-1], F[1:]))

        F_sum = np.sum(np.abs(F))
        F_angle = np.angle(F)
        norm_F = abs(F) / F_sum
        # Total number of Gaussian components in the spectrum
        K = 12

        F_gmm = gmm_model(K, norm_F)
        sigma_sq, omega = F_gmm.output_gmm()
        # print(sigma_sq, omega)
        # F_gmm.gmm_plot()
        # Initilize s1 and s2 with half of the Gaussians

        F_bin = F_gmm.F_bin

        s1_sigma_sq = sigma_sq[::2]
        s1_omega = omega[::2]
        s1_omega /= np.sum(s1_omega)
        K1 = int(len(s1_sigma_sq))


        s2_sigma_sq = sigma_sq[1::2]
        s2_omega = omega[1::2]
        s2_omega /= np.sum(s2_omega)
        K2 = int(len(s2_sigma_sq))

        # Calculate weighting probabilities gamma

        # noise
        sigma_sq_noise = np.random.random_sample()*0.5

        def Gaussian_log_product(sigma_sq1, sigma_sq2, norm_F):
            """ Use the log form to turn the multipulation to addition
            """
            N = len(norm_F)
            log_product = 0.0
            new_sigma_sq = sigma_sq1 + sigma_sq2 + sigma_sq_noise

            for i in range(N):
                log_product += -0.5*np.log(2*np.pi*new_sigma_sq) - norm_F[i]**2/(2.0*new_sigma_sq)
                # print(log_product, norm_F[i])
            return log_product
            

        log_gamma = np.zeros([K1, K2])

        for i in range(K1):
            for j in range(K2):
                log_product = Gaussian_log_product(s1_sigma_sq[i], s2_sigma_sq[j], F_bin)
                log_gamma[i][j] = np.log(s1_omega[i]) + np.log(s2_omega[j]) + log_product
        
        

        # Then normalize gamma
        max_log_gamma = np.max(log_gamma)
        log_gamma -= max_log_gamma
        # print(log_gamma)

        gamma = np.exp(log_gamma)
        gamma /= np.sum(gamma)
        # print(gamma)

        # Compute the posterior mean estimator
        N = len(norm_F)
        s1_F = np.zeros(N)
        s2_F = np.zeros(N)
        
        for i in range(K1):
            for j in range(K2):
                s1_F += gamma[i][j] * (s1_sigma_sq[i] / \
                (s1_sigma_sq[i]+s2_sigma_sq[j]+sigma_sq_noise)) * abs(F)
                # print(s1_spectrum[500], norm_F[500])
                # input()
                s2_F += gamma[i][j] * (s2_sigma_sq[j]/ \
                (s1_sigma_sq[i]+s2_sigma_sq[j]+sigma_sq_noise)) * abs(F)
        

        s1_spectrum[frame] = s1_F[N//2:]*np.exp(1j*F_angle[N//2:])
        s2_spectrum[frame] = s2_F[N//2:]*np.exp(1j*F_angle[N//2:])
    
                



        # plt.plot(s1_F])
        # plt.show()
        # plt.pause(0.5)
        # plt.clf()


    # plt.figure()
    # plt.imshow(abs(s1_spectrum).T, origin='lower', aspect='auto',
    #              interpolation='nearest')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.title('Signal 1 spectrum')
    # plt.show()


    # plt.figure()
    # plt.imshow(abs(s2_spectrum).T, origin='lower', aspect='auto',
    #              interpolation='nearest')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.title('Signal 2 spectrum')
    # plt.show()


    # Do the inverse transform and obtain the separated signals

    s1 = istft(s1_spectrum)
    s2 = istft(s2_spectrum)
    x = istft(X)

    fs, x1 = wavfile.read('./sound/male_sample.wav')
    fs, x2 = wavfile.read('./sound/female_sample.wav')

    plt.figure()
    plt.axis('scaled')

    plt.subplot(511)
    plt.plot(x)
    plt.title('Original signal mixture')

    plt.subplot(512)
    plt.plot(x1)
    plt.title('Male speaker signal')
    
    plt.subplot(513)    
    plt.plot(x2)
    plt.title('Female speaker signal')
    # plt.legend()

    plt.subplot(514)
    plt.plot(s1)
    plt.title('Separated male speaker')

    plt.subplot(515)
    plt.plot(s2)
    plt.title('Separated female speaker')

    # plt.show()



    # Calculate the SAR
    n1 = s1 - x1[:len(s1)]
    n2 = s2 - x2[:len(s2)]
    sar1 = 20*np.log10( np.sqrt(s1.dot(s1)) \
    / np.sqrt(n1.dot(n1))  )

    sar2 = 20*np.log10( np.sqrt(s2.dot(s2)) \
    / np.sqrt(n2.dot(n2))  )
    
    print(sar1, sar2)

