from stft import *
if __name__ == '__main__':
    fs, x = wavfile.read('./sound/speech-male-piano.wav')
    X = stft(x)

    N = len(x)
    t = np.linspace(0, N/fs, N)

    # Plot the magnitude spectrogram.
    plt.figure()
    plt.imshow(abs(X.T), origin='lower', aspect='auto',
                 interpolation='nearest')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

    # Compute the ISTFT.
    x_restored = istft(X)

    plt.figure()
    plt.plot(t, x)
    plt.xlabel('Time (seconds)')

    plt.figure()
    plt.plot(t[:len(x_restored+1)], x_restored)
    plt.xlabel('Time (seconds)')

