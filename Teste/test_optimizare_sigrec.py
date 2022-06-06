import numpy as np
import time


w_signal = np.random.rand(100, 1000)
overlap = 50

n = len(w_signal)  # number of windows
overlap = overlap / 100  # calc percentage
window_length = len(w_signal[0])  # window len

non_ov = int((1 - overlap) * window_length)  # non overlapping section of 2 windows
lenx = (n - 1) * non_ov + window_length  # len of signal to reconstruct. formula might be wrong.
delay = non_ov  # used to delay i'th window when creating the matrix that will be averaged

start = time.time()
w1 = np.zeros((n, lenx), dtype='float32')  # size = windows x signal_length
for i in range(n):
    crt1 = np.zeros(i * delay).tolist()
    crt1.extend(w_signal[i])
    crt1.extend(np.zeros(lenx - i * delay - window_length).tolist())

    w1[i] += crt1
stop = time.time()
print('time 1: ', stop - start)


start = time.time()
w2 = []
for i in range(n):
    crt2 = np.zeros(lenx, dtype='float32')
    crt2[i * delay:i * delay + len(w_signal[i])] = w_signal[i]
    w2.append(crt2)

w2 = np.asanyarray(w2, dtype='float32')
stop = time.time()
print('time 2:', stop - start)
print(np.allclose(w1, w2))
