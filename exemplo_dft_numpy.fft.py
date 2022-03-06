# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:27:10 2022

@author: rkrummenauer
"""
# importar os módulos e bibliotecas
import numpy as np
import matplotlib.pyplot as plt
#import os
#import IPython.display as ipd
#import librosa
#import librosa.display

from numpy.fft import fft, ifft, fftshift, fftfreq
#from scipy.fft import fft, ifft, fftshift, fftfreq

#%% Exemplo 1: uso básico

# geracao do sinal discreto x[n]=x(n.Ts)
f1,f2,f3 = 15, 42, 63        # freqs (Hz)
A1,A2,A3 = 1.0, 0.2, 0.3     # amplitudes
fs = 1000      # frequencia de amostragem (Hz)
dt = 1/fs      # periodo de amostragem (s)
tf = 20         # tempo final (s)

t = np.arange(0,tf,dt) # vetor de tempo
# geração do sinal
x = (A1*np.sin(2*np.pi*f1*t)
     + A2*np.sin(2*np.pi*f2*t)
     + A3*np.sin(2*np.pi*f3*t))

X = fft(x)
N = len(X)
print('len(X)=',N,'amostras na frequência')
print('len(x)=', len(x),'amostras no tempo')
f = np.arange(N)*fs/N

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(f, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Espectro de Amplitude |X(freq)|')
plt.xlim(0, 100)

plt.subplot(122)
plt.plot(t, x, 'b', label='original')
plt.plot(t, ifft(X), 'r', label='reconstrução com ifft')
plt.xlabel('Tempo (s)')
plt.ylabel('Sinal via ifft')
plt.legend()
plt.xlim(0, (1/f1))
plt.tight_layout()
plt.grid()

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(f, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Espectro de Amplitude |X(freq)|')
plt.title('Espectro de 0 a $f_{s}$')

plt.subplot(122)
plt.stem(fftshift(fftfreq(len(x),d=dt)), np.abs(fftshift(X))/N, 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Espectro de Amplitude |X(freq)|')
plt.title('Espectro de -$f_{s}/2$ a $f_{s}/2$ com fftshift')