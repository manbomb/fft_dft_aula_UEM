# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:30:58 2022

Este script trata da predição de uma série temporal.
O conjunto de dados é Activity Recognition, disponível no
repositório da UCI:
    https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer
    
    Os dados são de um acelerômetro triaxial montado no peito
    de 15 participantes, realizando 7 atividades cada.
    
    Attribute Information:

--- Data are separated by participant
--- Each file contains the following information
---- sequential number, x acceleration, y acceleration, z acceleration, label
--- Labels are codified by numbers
--- 1: Working at Computer
--- 2: Standing Up, Walking and Going updown stairs
--- 3: Standing
--- 4: Walking
--- 5: Going UpDown Stairs
--- 6: Walking and Talking with Someone
--- 7: Talking while Standing

@author: Rafael Krummenauer
"""

#%% Carregando os modulos e bibliotecas
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#import tensorflow as tf
#from tensorflow import keras

from scipy.fft import fft  # para analisar o espectro dos dados

#%% Carrega os dados de um participante do dataset
PATH = './data/datasets/activity_recognition_accelerometer/'

participante = '5' # são 15 ao total
FNAME = PATH+participante+'.csv'

# usa numpy.loadtxt() para ler o arquivo csv
data = np.loadtxt(FNAME,
           delimiter=',')
# data eh uma ndarray com todas as colunas do arquivo csv

# Seleciona apenas 1 das 7 atividades
index = data[:,4] == 5 # estamos tomando a atividade 5
data_sel = data[index,1:-2] 
''' já aproveitamos e tiramos a 1a coluna e a última
 A 1a coluna é apenas índice da sequência temporal.
 A última é o rótulo numérico (de 1 a 7) da atividade
 Ficamos apenas com 3 colunas: Ax, Ay e Az
'''
# Vamos trabalhar apenas com a sequência de aceleração do eixo x
data_Ax_raw = data_sel[:,0].flatten()

# visualiza dados crus
ind_t = np.arange(len(data_Ax_raw))
plt.figure()
plt.plot(ind_t,data_Ax_raw,label='Ax raw')
plt.legend()
plt.grid()
plt.rcParams.update({'font.size': 12})

#%% Analise dos dados, sabendo que fs = 52 Hz (frequência de amostragem)
'''
 Vamos inicialmente analisar o espectro do sinal para depois
 estabelecer uma frequência de corte e um filtro.
'''
fs = 52
N = len(data_Ax_raw)#1024
Xs = fft(data_Ax_raw)#,N) # FFT de N pontos
print('len(Xs)=',N,'amostras na frequência')
print('len(data_Ax_raw)=', len(data_Ax_raw),'amostras no tempo')
f = np.arange(N)*fs/N # frequências analisadas

plt.figure()
plt.stem(f, np.abs(Xs), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Espectro de Amplitude |X(freq)|')
plt.xlim(-1, 26)
#plt.ylim(0,120)

#%% Condicionamento/normalização dos dados
# escala entre -1 e 1
M = max(data_Ax_raw)
m = min(data_Ax_raw)
data_Ax_cond = -1.+2.*(data_Ax_raw-m)/(M-m)
# remove a média dos dados
data_mean = np.mean(data_Ax_cond)
data_Ax_cond = data_Ax_cond - data_mean

# visualiza dados normalizados
plt.figure()
plt.plot(ind_t,data_Ax_cond,label='Ax cond')
plt.legend()
plt.grid()

Xs_cond = fft(data_Ax_cond,N) # FFT de N pontos
plt.figure()
plt.stem(f, np.abs(Xs_cond), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Espectro de Amplitude condicionado |X(freq)|')
plt.xlim(0, 26)
#plt.ylim(0,120)