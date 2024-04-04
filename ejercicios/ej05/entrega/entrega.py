import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz


# Periodo de muestro de 1/16 uS
Ts = (1/16)*1E-5
Fs = 1/Ts
M = 16
N = 16
PRBS_N = 10

# print("Periodo de muestreo : {}".format(Ts))
# print("Tiempo de simbolo : {}".format(Ts*16))
# print("Frecuencia de muestreo : {}".format(Fs))

def generate_PRBS(length):
    # Generar una secuencia de bits aleatoria de longitud especificada
    random_bits = [random.choice([-1, 1]) for _ in range(length)]
    
    # Convierte los bits en una secuencia PRBS utilizando XOR
    prbs_sequence = []
    feedback = 0
    for bit in random_bits:
        feedback_bit = feedback & 1
        prbs_bit = feedback_bit ^ bit
        prbs_sequence.append(prbs_bit)
        feedback = (feedback >> 1) | ((feedback_bit ^ bit) << (length - 1))
    
    return prbs_sequence

def insert_zeros(signal, m):
    tmp = []
    for i in signal:
        tmp.append(i)
        for j in range(m - 1):
            tmp.append(0)
    return tmp


def pulso_cuadrado(comp = False, plot = False):
    
    T = 1
    fs = 1/16
    
    nT = int(T/fs)
    t = np.arange(0, T, fs)
    
    f = 1/T

    signal = np.zeros_like(t)
    signal[0:nT] = 1
    signal_p = np.tile(signal, int(len(t)/len(signal)))

    if comp == True:
        env = np.exp(1j * (2 * np.pi * f/2 * t))
        signal = signal * env
    
    if plot == True:
        plt.stem(t, np.abs(signal))
        if comp == True:
            plt.plot(t, signal.real, label="Real", color="r")
            plt.plot(t, signal.imag, label="Imag", color="g")
        plt.legend()
        plt.show()

    return signal,t


def pulso_triangular(comp = False, plot = False):
        
    T = 1
    fs = 1/16
    
    nT = int(T/fs)
    t = np.arange(0, T, fs)

    f = 1/T

    signal = np.zeros_like(t, dtype=np.complex128)
    signal[0:nT//2 + 1] = np.linspace(0, 1, nT//2 + 1)
    signal[nT//2 + 1:nT] = np.linspace(signal[nT//2-1], signal[1] , nT//2 - 1)

    if comp == True:
        env = np.exp(1j * (2 * np.pi * f * t))
        signal = signal * env
    
    if plot == True:
        plt.stem(t, np.abs(signal))
        if comp == True:
            plt.plot(t, signal.real, label="Real", color="r")
            plt.plot(t, signal.imag, label="Imag", color="g")
        plt.legend()
        plt.show()

    return signal,t

def pulso_seno(comp = False, plot = False):

    T = 1
    fs = 1/16
    
    nT = int(T/fs)
    t = np.arange(0, T, fs)

    f = 1/T

    signal = np.sin(2 * np.pi * (f/2) * t)

    if comp == True:
        env = np.exp(1j * (2 * np.pi * f * t))
        signal = signal * env
    
    if plot == True:
        plt.stem(t, np.abs(signal))
        if comp == True:
            plt.plot(t, signal.real, label="Real", color="r")
            plt.plot(t, signal.imag, label="Imag", color="g")
        plt.legend()
        plt.show()

    return signal,t

def pulso_cos_elevado(comp = False, plot = False):
    Tsim  = 1
    Ts    = 1/16
    
    t = np.arange(-Tsim*2, Tsim*2, Ts)
    Beta = 0.7

    f = 1/Tsim

    signal = 1/Tsim*np.sinc(t/Tsim)*np.cos(np.pi*Beta*t/Tsim)/(1-(2*Beta*t/Tsim)**2)

    signal[t==Tsim/2/Beta] = np.pi/4/Tsim*np.sinc(1/2/Beta)
    signal[t==-Tsim/2/Beta] = np.pi/4/Tsim*np.sinc(1/2/Beta)

    if comp == True:
        env = np.exp(1j * (2 * np.pi * f * t))
        signal = signal * env
    
    if plot == True:
        plt.stem(t, np.abs(signal))
        if comp == True:
            plt.plot(t, signal.real, label="Real", color="r")
            plt.plot(t, signal.imag, label="Imag", color="g")
        plt.legend()
        plt.show()

    return signal,t

def pulso_root_cos_elevado(comp = False, plot = False):
    Tsim  = 1
    Ts    = 1/16
    
    t = np.arange(-Tsim*2, Tsim*2, Ts)
    Beta = 0.7

    f = 1/Tsim

    a = np.sin(np.pi*t/Tsim*(1-Beta)) + 4*Beta*t/Tsim*np.cos(np.pi*t/Tsim*(1+Beta))
    b = np.pi*t/Tsim*(1-(4*Beta*t/Tsim)**2)
    signal = 1/Tsim*a/b

    signal[t==0] = 1/Tsim*(1+Beta*(4/np.pi-1))
    signal[t==Tsim/4/Beta] = Beta/Tsim/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/4/Beta)+(1-2/np.pi)*np.cos(np.pi/4/Beta))
    signal[t==-Tsim/4/Beta] = Beta/Tsim/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/4/Beta)+(1-2/np.pi)*np.cos(np.pi/4/Beta))

    if comp == True:
        env = np.exp(1j * (2 * np.pi * f * t))
        signal = signal * env
    
    if plot == True:
        plt.stem(t, np.abs(signal))
        if comp == True:
            plt.plot(t, signal.real, label="Real", color="r")
            plt.plot(t, signal.imag, label="Imag", color="g")
        plt.legend()
        plt.show()

    return signal,t


def convolucion(d, p):

    d_len = len(d)
    p_len = len(p)

    out_len = d_len + p_len - 1

    out = np.zeros(out_len)
    p = np.flip(p)


    for i in range(out_len):
        for j in range(p_len):
            if i - j >= 0 and i - j < d_len:
                out[i] += d[i - j] * p[j]
    
    out = out[(len(p) - 1)//2:]
    return out


def AddChannelNoise(signal: np.ndarray, SNR_dB: float) -> np.ndarray:
    """
    Adds additive white Gaussian noise (AWGN) to the signal.

    Parameters:
        signal (numpy.ndarray): Input signal.
        SNR_dB (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Signal with added noise.
    """
    signal_power = np.sum(signal**2) / len(signal)
    noise_power = signal_power / (10 ** (SNR_dB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise
    return noisy_signal

func = [pulso_cuadrado,
        pulso_triangular,
        pulso_seno,
        pulso_cos_elevado,
        pulso_root_cos_elevado]
p = []
x = []
c = []

def add_plot(ax,d,x,c,title):
    ax.stem(d, label="Deltas", linefmt='r-', markerfmt="ro")
    ax.stem(x, label="Output", markerfmt=' ')
    # ax.stem(c, label="Noisy", markerfmt='green')
    ax.set_xlabel('Muestras')
    ax.set_ylabel('Amplitud')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

def generar_canal(tipo):
    if tipo == 'delta_desfase':
        canal = np.zeros(M)
        canal[1] = 1
        return canal
    if tipo == 'pasabajos':
        fs = 32
        fc = 8
        taps = 5
        filtro_pb = firwin(taps, fc, fs=fs)
        w, h = freqz(filtro_pb)
        canal = np.fft.ifft(filtro_pb)
        return canal
    else:
        return 0


def simular(canal : str = 'ideal'):
    fig, ax = plt.subplots(5,3)

    b = generate_PRBS(PRBS_N)
    d = insert_zeros(b, M)

    p = []
    x = []
    h = []
    r = []
    c = []

    h = generar_canal(tipo = canal)

    for i,f in enumerate(func):
        s, t = f(comp=True)
        p.append(s)
        x.append(convolucion(d, np.abs(p[i])))

        if canal != 'ideal':
            print("Canal : {}".format(canal))
            r.append(convolucion(x[i],h))
        else:
            print("Canal ideal, no agrega distorsion")
            r.append(x[i])

        # Agregar ruido
        c.append(AddChannelNoise(r[i], 5))
        add_plot(ax[i,0],d,x[i],c[i],title=canal)
        add_plot(ax[i,1],d,r[i],c[i],title=canal)

    plt.show()


# ax[0,0].set_title('PRBS')
# ax[0,1].set_title('ESPECTRO PRBS')
# ax[0,2].set_title('ESPECTRO PRBS-NOISY')

# for i in range(4):
#     ax[i,0].stem(d, label="Deltas", linefmt='r-', markerfmt="ro")
#     ax[i,0].stem(x[i], label="Output", markerfmt=' ')
#     ax[i,0].stem(c[i], label="Noisy", markerfmt='green')
#     ax[i,0].set_xlabel('Muestras')
#     ax[i,0].set_ylabel('Amplitud')
#     ax[i,0].legend()
#     ax[i,0].grid(True)

# fft_x = [20*np.log10((np.abs(np.fft.fft(xi)))) for xi in x] 
# fft_c = [20*np.log10((np.abs(np.fft.fft(ci)))) for ci in c] 

# freq = np.fft.fftfreq(len(x[0]), 1/16)
# positive = np.where(freq > 0)

# for i in range(4):
#     ax[i,1].plot(freq[positive], fft_x[i][positive], label="PRBS")
#     ax[i,1].set_xlabel('Frequency (MHz)')
#     ax[i,1].set_ylabel('Magnitude')
#     ax[i,1].legend()
#     ax[i,1].grid(True)

#     ax[i,2].plot(freq[positive], fft_c[i][positive], label="Noisy")
#     ax[i,2].set_xlabel('Frequency (MHz)')
#     ax[i,2].set_ylabel('Magnitude')
#     ax[i,2].legend()
#     ax[i,2].grid(True)

# plt.tight_layout()
# plt.show()

# fix,ax1 = plt.subplots(4)
# for i in range(4):
#     ax1[i].stem(d, label="Deltas", linefmt='r-', markerfmt="ro")
#     ax1[i].stem(x[i], label="Output")
#     ax1[i].stem(c[i], label="Noisy",markerfmt='green')
#     ax1[i].set_xlabel('Muestras')
#     ax1[i].set_ylabel('Amplitud')
#     ax1[i].legend()
#     ax1[i].grid(True)

# plt.tight_layout()
# plt.show()