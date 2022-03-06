# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:36:07 2021

Exemplo de FFT com o modulo numpy.fft:
analise espectral de 2 senoides com ruido

@author: Rafael
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
import math


#%% geracao do sinal discreto x[n]=x(n.Ts)
f1 = 15        # f1 (Hz)
f2 = 40        # f2 (Hz)
A1 = 1.5       # amplitude sinal 1
A2 = 1.0       # amplitude sinal 2
fs = 1000       # frequencia de amostragem (Hz)
tf = 10        # tempo final
dt = 1/fs      # periodo de amostragem

t = np.arange(0,tf,dt) # vetor de tempo

# geração do sinal
x = (A1*np.cos(2*np.pi*f1*t) + A2*np.cos(2*np.pi*f2*t)
      +1.0*np.random.randn(len(t)))

plt.plot(t,x)
plt.title('2 tons com ruído')
plt.xlabel('tempo')
plt.ylabel('Amplitude')

#%% Uso da FFT para calcular a DFT

# funcao para retornar a proxima potencia de 2
def nextpow2(n):  
    return 1 if n == 0 else 2**math.ceil(math.log2(n))

M = len(x)              # tamanho do sinal original
N = nextpow2(M)         # tamanho da DFT (potencia de 2)
X = fft(x,N)         # calculo da DFT direta pela FFT
f = np.arange(0,N)*(fs/N)     # frequencias calculadas
PowerSpectrum = X*np.conj(X)/N   # potencia da DFT

df = f[1]-f[0]         # delta entre frequencias amostradas
print('\nInformações da FFT:')
print('   df = {:.3f} Hz'.format(df))
print('   Tamanho do sinal: M = {}'.format(M))
print('   Tamanho da DFT: N = {}'.format(N))
print('   Tamanho do Zero-Padding: {}'.format(N-M))

#%% plot do espectro da DFT (tambem chamado de Periodograma)
plt.figure()
plt.plot(f,PowerSpectrum)
plt.xlabel('Frequencia (Hz)')
plt.ylabel('Potencia')
plt.title('Periodograma (Espectro de Potência)')

#%%  Centrar em f=0 com fftshift (fazendo uso da propriedade da periodicidade da DFT)
X0 = fftshift(X)          # rearranja (gira) valores das componentes 
f0 = np.arange(-N/2,N/2)*(fs/N)  # vetor f centrado em zero
#PSpectrum0 = X0*np.conj(X0)/N   # potencia centrada em f=0
PowerSpectrum0 = np.power(np.abs(X0),2)/N        # potencia centrada em f=0

plt.figure()
plt.stem(f0,PowerSpectrum0,'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Frequência (Hz)')
plt.ylabel('Potência')
plt.title('Periodograma centrado em f=0 Hz')

#%% FFT inversa
plt.figure(figsize = (12, 6))
plt.subplot(221)

plt.stem(f, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude |X(freq)|')
plt.xlim(0, f2+10.)

plt.subplot(222)
x_ifft = ifft(X)
plt.plot(t, x_ifft[0:M], 'r',label='ifft(X)')
plt.plot(t,x,'g',label='x[n] original')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

plt.subplot(223)
plt.plot(t,x-x_ifft[0:M],'k',label='erro na inversão')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

plt.subplot(224)
plt.stem(f,20*np.log10(np.abs(fft(x-x_ifft[0:M],N))),'b', \
         label='Espectro de Amplitude do erro em dB', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()